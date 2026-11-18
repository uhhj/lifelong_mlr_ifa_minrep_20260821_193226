"""Desktop demonstration recorder for the current PiPER binding worker."""

import base64
import json
import time
import tkinter as tk
from tkinter import ttk

import cv2

from cameras import CameraStream, configured_cameras
from capture_core import DATA, ROOT, EpisodeWriter, audit_episode
from timeline import READ_WINDOW_NS, TimelineTail, following, read_status


BINDINGS = ROOT/'official_ui_teleop_v1'/'binding_runs'
TASKS = (
    ('B1', '夹起木块，放进左侧蓝碟'),
    ('B2', '夹起木块，放进右侧浅色碟'),
    ('B3', '把木块沿桌面推到左侧黄色区域'),
    ('I1', '夹起木块，放进中间粉碟'),
    ('I2', '把木块沿桌面推到右侧紫色区域'),
    ('I3', '夹起木块，放进中央红色精确放置区'),
)


def paired_frames(cameras, start, end, last_sequences):
    external, wrist = ([
        f for f in cameras[role] if start <= f['host_monotonic_ns'] <= end
        and f['sequence'] > last_sequences.get(role, 0)
    ] for role in ('external','wrist'))
    for front in external:
        if not wrist:
            break
        closest = min(range(len(wrist)),
                      key=lambda i:abs(wrist[i]['host_monotonic_ns']-front['host_monotonic_ns']))
        hand = wrist[closest]
        if abs(hand['host_monotonic_ns']-front['host_monotonic_ns']) <= READ_WINDOW_NS:
            yield {'external':front, 'wrist':hand}
            del wrist[:closest+1]


class RecorderWindow:
    def __init__(self, window):
        self.window = window
        self.streams = {}
        self.images = {}
        self.tail = None
        self.writer = None
        self.finish_request = None
        self.last_command = None
        self.last_preview = 0
        self.status = tk.StringVar(value='正在识别相机')
        self.message = tk.StringVar(value='录制目录 '+str(DATA))
        window.title('PiPER 同步录制')
        window.protocol('WM_DELETE_WINDOW',self.close)
        ttk.Label(window,textvariable=self.status,padding=10).pack(anchor='w')
        views = ttk.Frame(window,padding=8)
        views.pack(fill='x')
        self.previews = {}
        for col, (role,label) in enumerate((('external','D435i 俯视'),('wrist','D405 腕部'))):
            ttk.Label(views,text=label).grid(row=0,column=col)
            preview = ttk.Label(views,text='等待图像',width=35)
            preview.grid(row=1,column=col,padx=6)
            self.previews[role] = preview
        form = ttk.Frame(window,padding=8)
        form.pack(fill='x')
        ttk.Label(form,text='任务').grid(row=0,column=0)
        ttk.Label(form,text='木块起点').grid(row=0,column=1)
        ttk.Label(form,text='用途').grid(row=0,column=2)
        self.task = ttk.Combobox(form,state='readonly',width=37,
                                 values=[f'{task}  {instruction}' for task,instruction in TASKS])
        self.task.grid(row=1,column=0,padx=5)
        self.task.current(0)
        self.initial = ttk.Combobox(form,state='readonly',width=10,values=['中心','前','后','左','右'])
        self.initial.grid(row=1,column=1,padx=5)
        self.initial.current(0)
        self.purpose = ttk.Combobox(form,state='readonly',width=14,values=['试采','专家示教'])
        self.purpose.grid(row=1,column=2,padx=5)
        self.purpose.current(0)
        buttons = ttk.Frame(window,padding=8)
        buttons.pack(fill='x')
        self.start_button = ttk.Button(buttons,text='开始录制',command=self.start,state='disabled')
        self.start_button.pack(side='left',padx=5)
        self.end_buttons = []
        for label,outcome in (('成功结束','success'),('失败结束','failure'),('中止并保存','aborted')):
            button = ttk.Button(buttons,text=label,command=lambda x=outcome:self.end(x),state='disabled')
            button.pack(side='left',padx=5)
            self.end_buttons.append(button)
        ttk.Button(buttons,text='重连相机',command=self.open_cameras).pack(side='right',padx=5)
        ttk.Label(window,textvariable=self.message,wraplength=820,padding=8).pack(anchor='w')
        self.open_cameras()
        self.window.after(33,self.tick)

    def open_cameras(self):
        if self.writer:
            return
        for stream in self.streams.values():
            stream.close()
        self.streams = {}
        try:
            self.streams = {role:CameraStream(conf) for role,conf in configured_cameras().items()}
        except (ValueError,OSError) as exc:
            self.message.set(str(exc))

    def refresh(self):
        if self.writer:
            current = self.tail.directory
        else:
            runs = sorted((p for p in BINDINGS.iterdir() if p.is_dir()),reverse=True) if BINDINGS.is_dir() else []
            current = next((p for p in runs[:1] if following(read_status(p))),None)
        if self.tail and self.tail.directory != current:
            self.tail.close()
            self.tail = None
        if current and not self.tail and (current/'can_timeline.jsonl').is_file():
            self.tail = TimelineTail(current)
        if not self.tail:
            return False
        for event,echo in self.tail.read():
            if self.writer and self.writer.started <= event['mono_ns'] and (
                    not self.finish_request or event['mono_ns'] <= self.finish_request[1]):
                self.writer.append_raw(event)
                if echo:
                    self.writer.append_commands([echo])
        return following(read_status(self.tail.directory))

    def available(self, now):
        if self.tail is None or set(self.streams) != {'external','wrist'}:
            return False
        timeline = self.tail.timeline
        if not 0 <= now-timeline.watermark < 300_000_000:
            return False
        frames = [stream.latest() for stream in self.streams.values()]
        return (all(frame and 0 <= now-frame['host_monotonic_ns'] < 120_000_000 for frame in frames)
                and timeline.sample_at(timeline.watermark) is not None)

    def start(self):
        if self.writer or not self.tail or not following(read_status(self.tail.directory)):
            return
        now = time.monotonic_ns()
        if not self.available(now):
            self.message.set('请等待两路图像、从臂反馈和关节及夹爪的有效目标。')
            return
        task,instruction = TASKS[self.task.current()]
        initial_command = self.tail.timeline.sample_at(self.tail.timeline.watermark)[1]
        try:
            self.writer = EpisodeWriter(dict(
                task_id=task,task=instruction,initial_condition_id=self.initial.get(),
                purpose='commissioning' if self.purpose.current()==0 else 'demonstration',
                episode_type='human_teleoperation',started_monotonic_ns=now,
                source_binding={'run_id':self.tail.directory.name,'timeline':'can_timeline.jsonl'},
                camera_config={role:camera.config for role,camera in self.streams.items()},
                scene='PiPER 固定三碟三色区域', initial_confirmed_target=initial_command))
            self.last_command = (initial_command['joint_sent_mono_ns'],initial_command['gripper_sent_mono_ns'])
            self.message.set('正在录制。撤爪、确认终态后选择结束结果。')
            for widget in (self.task,self.initial,self.purpose):
                widget.configure(state='disabled')
            for button in self.end_buttons:
                button.configure(state='normal')
        except (OSError,ValueError) as exc:
            self.message.set(str(exc))

    def record_pairs(self):
        timeline = self.tail.timeline
        cutoff = timeline.watermark-READ_WINDOW_NS
        if self.finish_request:
            cutoff = min(cutoff,self.finish_request[1])
        cameras = {role:stream.frames() for role,stream in self.streams.items()}
        for pair in paired_frames(cameras,self.writer.started,cutoff,self.writer.last_camera_seq):
            stamp = max(frame['host_monotonic_ns'] for frame in pair.values())
            if self.writer.last_ns is not None and stamp <= self.writer.last_ns:
                continue
            item = timeline.sample_at(stamp)
            if item is None:
                continue
            state,command = item
            targets = command['joint_sent_mono_ns'],command['gripper_sent_mono_ns']
            command['joint_held'] = targets[0] == self.last_command[0]
            command['gripper_held'] = targets[1] == self.last_command[1]
            if self.writer.append(state,command,pair,stamp):
                self.last_command = targets

    def end(self,outcome):
        if self.writer and self.finish_request is None:
            self.finish_request = (outcome,time.monotonic_ns())
            for button in self.end_buttons:
                button.configure(state='disabled')

    def finish(self,outcome,stamp):
        writer,self.writer = self.writer,None
        self.finish_request = None
        try:
            target = writer.finish(outcome,stamp)
            audit = audit_episode(target)
            summary = json.loads((target/'summary.json').read_text())
            message = f'已保存 {target.name} · {audit["samples"]} 组 · 最大覆盖间隔 {summary["max_coverage_gap_ms"]:.1f} ms'
            if not summary['sampling_checks_pass']:
                message += ' · 同步检查未通过'
            self.message.set(message)
        except (OSError,ValueError) as exc:
            self.message.set(f'保存异常，文件保留在 {writer.folder} · {exc}')
        finally:
            for widget in (self.task,self.initial,self.purpose):
                widget.configure(state='readonly')
            for button in self.end_buttons:
                button.configure(state='disabled')

    def tick(self):
        try:
            active = self.refresh()
            now = time.monotonic_ns()
            ready = active and self.available(now)
            self.start_button.configure(state='normal' if ready and not self.writer else 'disabled')
            if self.writer:
                self.record_pairs()
                if self.finish_request:
                    outcome,cutoff = self.finish_request
                    if self.tail.timeline.watermark >= cutoff+READ_WINDOW_NS:
                        self.finish(outcome,cutoff)
                    elif not active or now-cutoff > 1_000_000_000:
                        self.finish('aborted',cutoff)
                elif not active:
                    self.finish('aborted',now)
            if now-self.last_preview > 300_000_000:
                self.show_preview(ready)
                self.last_preview = now
        except (OSError,ValueError,KeyError) as exc:
            if self.writer:
                self.finish('aborted',time.monotonic_ns())
            if self.tail:
                self.tail.close()
                self.tail = None
            self.message.set(f'采集检查：{exc}')
        self.window.after(33,self.tick)

    def show_preview(self,ready):
        self.status.set('录制中' if self.writer else '可开始录制' if ready else '等待双路图像和有效的遥操目标')
        for role,stream in self.streams.items():
            frame = stream.latest()
            if frame and time.monotonic_ns()-frame['host_monotonic_ns'] < 500_000_000:
                ok,picture = cv2.imencode('.png',cv2.resize(frame['image'],(400,300)))
                if ok:
                    photo = tk.PhotoImage(data=base64.b64encode(picture.tobytes()))
                    self.images[role] = photo
                    self.previews[role].configure(image=photo,text='')
            else:
                self.previews[role].configure(image='',text=stream.error or '图像已过期')

    def close(self):
        if self.writer:
            self.finish('aborted',time.monotonic_ns())
        if self.tail:
            self.tail.close()
        for stream in self.streams.values():
            stream.close()
        self.window.destroy()


if __name__ == '__main__':
    window = tk.Tk()
    RecorderWindow(window)
    window.mainloop()
