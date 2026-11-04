$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $localTectonic = Join-Path $PSScriptRoot '../../local_assets/tectonic/tectonic.exe'
    if (Test-Path -LiteralPath $localTectonic) {
        & $localTectonic -X compile main.tex --keep-logs --keep-intermediates
    } elseif (Get-Command tectonic -ErrorAction SilentlyContinue) {
        tectonic -X compile main.tex --keep-logs --keep-intermediates
    } elseif (Get-Command latexmk -ErrorAction SilentlyContinue) {
        latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
    } else {
        throw 'Install a TeX distribution, or upload the source ZIP to Overleaf (main.tex, pdfLaTeX).'
    }
    if ($LASTEXITCODE -ne 0) { throw 'LaTeX compilation failed.' }
} finally { Pop-Location }
