const fs=require('fs'),path=require('path'),crypto=require('crypto');
const sharp=require('C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1104-2876-0_26.1104.2876.0_x64__3k8sg7r9htsxt/dependencies/node/node_modules/sharp');
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
async function main(){
 const svg=path.join(__dirname,'DEBR_C_journal.svg');
 const preview=await sharp(svg).png().toBuffer();fs.writeFileSync(path.join(__dirname,'DEBR_C_journal_preview.png'),preview);
 const png=path.join(__dirname,'DEBR_C_journal.png');await sharp(svg,{density:144}).png().toFile(png);
 const small=await sharp(png).resize(1800,860).png().toBuffer();
 const labels=Buffer.from('<svg width="3660" height="70" xmlns="http://www.w3.org/2000/svg"><rect width="3660" height="70" fill="white"/><text x="900" y="44" font-family="Times New Roman" font-size="30" text-anchor="middle" fill="#193247">Editable SVG</text><text x="2760" y="44" font-family="Times New Roman" font-size="30" text-anchor="middle" fill="#193247">PNG from the identical SVG master</text></svg>');
 await sharp({create:{width:3660,height:930,channels:4,background:'white'}}).composite([{input:preview,left:0,top:0},{input:small,left:1860,top:0},{input:labels,left:0,top:860}]).png().toFile(path.join(__dirname,'SVG_PNG_comparison.png'));
 const r={svg_sha256:sha(svg),png_sha256:sha(png),svg_canvas:[1800,860],png_pixels:[3600,1720],single_master:true};
 fs.writeFileSync(path.join(__dirname,'RENDER_RECEIPT.json'),JSON.stringify(r,null,2)+'\n');console.log(JSON.stringify(r));
}
main().catch(e=>{console.error(e);process.exit(1)});
