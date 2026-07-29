param(
    [string]$Destination = "data/cache/upstream/xarr_all_1000.nc"
)

$ErrorActionPreference = "Stop"
$oid = "8a68c3f6b57c6aac641fc3ebb0f19ba406c1426aecb55526cbc855ca12b84eee"
$relativeObject = Join-Path ".git/lfs/objects" (Join-Path $oid.Substring(0, 2) (Join-Path $oid.Substring(2, 2) $oid))

git lfs fetch origin 8f11565 --include="notebooks/deepjr_training_data/xarr_all_1000.nc"
if (-not (Test-Path -LiteralPath $relativeObject)) {
    throw "Git LFS object was not downloaded: $relativeObject"
}

$destinationPath = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $Destination))
$repositoryPath = [System.IO.Path]::GetFullPath((Get-Location).Path)
if (-not $destinationPath.StartsWith($repositoryPath, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Destination must stay inside the repository: $destinationPath"
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destinationPath) | Out-Null
Copy-Item -LiteralPath $relativeObject -Destination $destinationPath
$hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $destinationPath).Hash.ToLowerInvariant()
if ($hash -ne $oid) {
    throw "SHA-256 mismatch after copy: $hash"
}
Write-Output "Prepared $destinationPath ($hash)"
