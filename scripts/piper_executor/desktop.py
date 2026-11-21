"""Onsite operator panel. Opening this window only creates UI widgets."""
import math
import select
import socket
import subprocess
import sys
import tkinter as tk
from tkinter import messagebox, ttk

from runner import HERE, TASKS, receive, send


class Panel:
    def __init__(self, root):
        self.root=root
        root.title("PiPER 策略执行台")
        self.link=self.proc=None
        self.active=self.closing=self.done=self.pending=False
        self.task=tk.StringVar(value=TASKS[0])
        self.status=tk.StringVar(value="就绪。选择任务后，先开始只读预测。")
        self.values=tk.StringVar()
        box=ttk.Frame(root,padding=18)
        box.pack(fill="both",expand=True)
        self.choice=ttk.Combobox(box,textvariable=self.task,values=TASKS,state="readonly",width=34)
        self.choice.pack(fill="x")
        self.preview=ttk.Button(box,text="开始只读预测",command=self.open)
        self.start=ttk.Button(box,text="开始现场执行",command=self.begin)
        self.stop=tk.Button(box,text="停止 / STOP",fg="white",bg="#a51f24",command=lambda:self.end("stop"))
        for btn in (self.preview,self.start,self.stop): btn.pack(fill="x",pady=3)
        outcomes=ttk.Frame(box)
        outcomes.pack(fill="x")
        self.success=ttk.Button(outcomes,text="成功结束并 STOP",command=lambda:self.end("success"))
        self.failed=ttk.Button(outcomes,text="失败结束并 STOP",command=lambda:self.end("failed"))
        for btn in (self.success,self.failed): btn.pack(side="left",expand=True,fill="x")
        ttk.Label(box,textvariable=self.status,wraplength=510).pack(anchor="w",pady=6)
        ttk.Label(box,textvariable=self.values,justify="left").pack(anchor="w")
        self.log=tk.StringVar()
        ttk.Label(box,textvariable=self.log).pack(anchor="w")
        self.controls(False)
        root.protocol("WM_DELETE_WINDOW",self.close)
        root.after(50,self.poll)

    def controls(self,ready):
        running=self.proc is not None
        self.choice.configure(state="disabled" if running else "readonly")
        self.preview.configure(state="disabled" if running else "normal")
        self.start.configure(state="normal" if running and ready and not self.active and not self.pending else "disabled")
        self.stop.configure(state="normal" if running else "disabled")
        for btn in (self.success,self.failed): btn.configure(state="normal" if self.active else "disabled")

    def open(self):
        if self.proc: return
        a,b=socket.socketpair(type=socket.SOCK_SEQPACKET)
        try:
            with b:
                self.proc=subprocess.Popen(["/usr/bin/python3","-B",str(HERE/"runner.py"),"guard",
                       str(b.fileno()),str(TASKS.index(self.task.get())),sys.executable],pass_fds=(b.fileno(),))
        except OSError as e:
            a.close()
            self.status.set(str(e))
            return
        self.link=a
        self.link.setblocking(False)
        self.done=self.active=self.pending=False
        self.values.set("")
        self.log.set("")
        self.status.set("正在检查相机与 CAN 接口")
        self.controls(False)

    def command(self,op):
        try: send(self.link,dict(op=op))
        except OSError:
            self.link.close()
            self.link=None
            self.status.set("控制进程通信中断。若操作臂在运动，请使用现场独立急停。")

    def begin(self):
        if not self.link or self.pending or self.active: return
        if messagebox.askokcancel("现场确认",self.task.get()+"\n\n请确认主臂已物理断开，静止抽动已排除，从臂已现场准备为普通 CAN 关节位置速度模式，独立急停在手边。\n\n确定后将下发实际运动目标。",parent=self.root):
            if self.link and not self.done:
                self.pending=True
                self.controls(False)
                self.command("start")

    def end(self,op):
        if self.link and not self.done:
            self.command(op)
            self.pending=True
            self.controls(False)
            self.status.set("正在结束本轮")

    def close(self):
        if self.proc:
            self.closing=True
            self.end("close")
        else: self.root.destroy()

    def show(self,msg):
        if "log" in msg:
            label="本轮日志  "+msg["log"]
            if "capture_dir" in msg:
                label+="\n双相机同步记录  "+msg["capture_dir"]+"/inputs.jsonl"
            self.log.set(label)
        if "active" in msg:
            self.active=msg["active"]
            if self.active: self.pending=False
        if "pred" in msg:
            def display(q):
                return "  ".join(f"{math.degrees(v):.1f}°" for v in q[:6])+f"    {q[6]:.1f} mm"
            self.values.set("当前  "+display(msg["q"])+"\n预测  "+display(msg["pred"]))
            if not self.active and not self.pending and not self.status.get().startswith("执行条件"):
                self.status.set("只读预测中。机械臂尚未由策略控制。")
            self.controls(True)
        if "text" in msg: self.status.set(msg["text"])
        if "error" in msg:
            self.status.set("执行条件未满足："+msg["error"])
            self.pending=False
            self.controls(True)
        if msg.get("done"):
            self.done=True
            self.active=False
            self.controls(False)

    def poll(self):
        if self.link and not self.done:
            try:
                while select.select([self.link],[],[],0)[0]:
                    self.show(receive(self.link))
                    if self.done: break
                if not self.done: self.command("beat")
            except (EOFError,OSError):
                self.done=True
                self.status.set("监控连接中断。若操作臂在运动，请使用现场独立急停。")
        if self.proc and self.proc.poll() is not None:
            if self.link:
                self.link.close()
            self.link=self.proc=None
            self.active=self.pending=False
            self.controls(False)
            if self.closing:
                self.root.destroy()
                return
        self.root.after(50,self.poll)


if __name__=="__main__":
    root=tk.Tk()
    Panel(root)
    root.mainloop()
