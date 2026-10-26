import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
import * as b from '../build_slides.mjs';
const R=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const SKILL='C:/Users/Lenovo/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11618/skills/presentations';
const presentation=Presentation.create({slideSize:{width:1280,height:720}});presentation.slides.add();
await b.first({presentation});await b.middle({presentation});await b.last({presentation});
await b.realframes({presentation,input:{assetDir:R+'/assets_private'}});await b.training({presentation,input:{curvePath:R+'/training_curves_S41.png'}});
for(let i=0;i<6;i++){const slide=presentation.slides.getItem(i);const out=await presentation.export({slide,format:'png',scale:1});await fs.writeFile(R+'/output/slide-'+(i+1)+'.png',new Uint8Array(await out.arrayBuffer()));const lay=await slide.export({format:'layout'});await fs.writeFile(R+'/.build/slide-'+(i+1)+'.layout.json',await lay.text());}
const candidate=R+'/.build/candidate.pptx';await (await PresentationFile.exportPptx(presentation)).save(candidate);
const {finalizePresentation}=await import(pathToFileURL(SKILL+'/container_tools/artifact_tool_utils.mjs').href);
const result=await finalizePresentation({workspaceDir:R,candidatePath:candidate,finalPath:R+'/output/前期工作_MLR-IFA_RA_ALIGN_S_实证版.pptx',pythonExecutable:'C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:SKILL+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:SKILL+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit','--require-native-table-slide','3','--require-native-table-slide','4'],explicitTotalSlideCount:6,requiredNativeTableOwnerSlides:[3,4],requiredNativeChartOwnerSlides:[],fontPolicy:{basis:'design',families:['Microsoft YaHei','Consolas']},verifyArtifactToolImport:true,receiptPath:R+'/.build/validation.json'});
console.log(JSON.stringify(result));
