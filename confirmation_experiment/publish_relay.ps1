param(
    [Parameter(Mandatory=$true)][string]$Mirror,
    [Parameter(Mandatory=$true)][string]$OutputDirectory,
    [string]$Identity = 'C:\Users\Lenovo\.ssh\codex_jq1_9gpu',
    [string]$SshHost = 'wzp2026@172.21.105.127'
)
$ErrorActionPreference = 'Stop'
$branch = 'research/fixed-kappa-confirmation-v2'
$remoteRoot = '/media/dm/Elements/wzp2026/lifelong_mlr_ifa_minrep'
$remotePython = '/home/dm/miniconda3/envs/wzp_mlr_tmp/bin/python'
$sshOptions = @('-i',$Identity,'-o','BatchMode=yes','-o','ConnectTimeout=10')
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$statusPath = Join-Path $OutputDirectory 'publication_relay_status.json'
function Set-RelayStatus([string]$phase, [string]$detail) {
    @{ phase=$phase; detail=$detail; pid=$PID; time=(Get-Date -Format o) } |
        ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8
}
Set-RelayStatus 'WAITING' 'Waiting for the six-run queue to commit its final report.'
try {
    while ($true) {
        $response = & ssh @sshOptions $SshHost "cd $remoteRoot && $remotePython -m confirmation_experiment.publication" 2>$null
        if ($LASTEXITCODE -ne 0) { Start-Sleep -Seconds 60; continue }
        $state = ($response -join "`n") | ConvertFrom-Json
        if ($state.phase -eq 'FAILED') { throw ('Remote integrity/execution failure: '+$state.error) }
        if ($state.phase -in @('PUBLISHED','COMPLETE')) { Set-RelayStatus 'COMPLETE' $state.results_sha; break }
        if ($state.phase -ne 'READY') { Start-Sleep -Seconds 60; continue }
        if ($state.results_sha -notmatch '^[0-9a-f]{40}$') { throw 'Invalid report commit SHA' }
        $bundlePath = Join-Path $OutputDirectory 'RESULTS.bundle'
        & scp @sshOptions "${SshHost}:$remoteRoot/reports/fixed_kappa_confirmation_v2/RESULTS.bundle" $bundlePath
        if ($LASTEXITCODE -ne 0) { throw 'Cannot retrieve report bundle' }
        & git -C $Mirror fetch $bundlePath "refs/heads/${branch}:refs/heads/${branch}"
        if ($LASTEXITCODE -ne 0) { throw 'Cannot import report commit' }
        $actual = (& git -C $Mirror rev-parse "refs/heads/$branch").Trim()
        if ($actual -ne $state.results_sha) { throw 'Bundle commit differs from queue witness' }
        & git -C $Mirror -c credential.helper= -c 'credential.helper=!gh auth git-credential' push origin "refs/heads/${branch}:refs/heads/${branch}"
        if ($LASTEXITCODE -ne 0) { throw 'Authenticated push failed' }
        $line = & git -C $Mirror -c credential.helper= -c 'credential.helper=!gh auth git-credential' ls-remote origin "refs/heads/$branch"
        if ($LASTEXITCODE -ne 0 -or ($line -split '\s+')[0] -ne $state.results_sha) { throw 'Remote branch verification failed' }
        & ssh @sshOptions $SshHost "cd $remoteRoot && $remotePython -m confirmation_experiment.publication --ack-sha $($state.results_sha)"
        if ($LASTEXITCODE -ne 0) { throw 'Cannot acknowledge verified publication' }
        Set-RelayStatus 'COMPLETE' $state.results_sha
        break
    }
} catch {
    Set-RelayStatus 'FAILED' $_.Exception.Message
    throw
}
