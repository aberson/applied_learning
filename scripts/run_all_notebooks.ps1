# Re-run every topic notebook headless as a runnability gate.
# Executes in place so committed notebooks keep their embedded graphics.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$nbs = Get-ChildItem -Path (Join-Path $root "topics") -Recurse -Filter *.ipynb |
    Where-Object { $_.FullName -notmatch "\.ipynb_checkpoints" }

$failed = @()
foreach ($nb in $nbs) {
    Write-Host ">> executing: $($nb.FullName)"
    uv run jupyter nbconvert --to notebook --execute --inplace "$($nb.FullName)"
    if ($LASTEXITCODE -ne 0) { $failed += $nb.FullName }
}

if ($failed.Count -gt 0) {
    Write-Host ""
    Write-Host "FAILED ($($failed.Count)):"
    $failed | ForEach-Object { Write-Host "  $_" }
    exit 1
}
Write-Host ""
Write-Host "All $($nbs.Count) notebooks executed cleanly."
