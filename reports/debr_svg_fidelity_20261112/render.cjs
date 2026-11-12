const path=require('path');
const sharp=require('C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1104-2876-0_26.1104.2876.0_x64__3k8sg7r9htsxt/dependencies/node/node_modules/sharp');
sharp(path.join(__dirname,'DEBR_A_refined_faithful.svg')).png().toFile(path.join(__dirname,'DEBR_A_refined_faithful_preview.png')).then(()=>console.log('Rendered')).catch(e=>{console.error(e);process.exit(1);});
