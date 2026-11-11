const sharp = require('C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1104-2876-0_26.1104.2876.0_x64__3k8sg7r9htsxt/dependencies/node/node_modules/sharp');
const path = require('path');
sharp(path.join(__dirname, 'DEBR_A_editable.svg')).png().toFile(path.join(__dirname, 'DEBR_A_editable_preview.png')).then(()=>console.log('Rendered SVG preview')).catch(e=>{console.error(e);process.exit(1);});
