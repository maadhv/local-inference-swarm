param(
    [string]$Version = "0.1.0"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$destination = Join-Path $root "dist\localswarm-$Version-source.zip"
New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
if (Test-Path -LiteralPath $destination) { Remove-Item -LiteralPath $destination -Force }
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$stream = [System.IO.File]::Open($destination, [System.IO.FileMode]::CreateNew)
try {
    $archive = [System.IO.Compression.ZipArchive]::new($stream, [System.IO.Compression.ZipArchiveMode]::Create)
    try {
        $files = @("README.md", "pyproject.toml", "MANIFEST.in", ".gitignore", "scripts/package-source.ps1") | ForEach-Object { Join-Path $root $_ }
        $files += Get-ChildItem -LiteralPath (Join-Path $root "src\localswarm") -File -Recurse | Where-Object { $_.FullName -notmatch "[\\/]__pycache__([\\/]|$)" } | ForEach-Object { $_.FullName }
        foreach ($file in $files) {
            $relative = [System.IO.Path]::GetRelativePath($root, $file).Replace("\", "/")
            [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($archive, $file, $relative, [System.IO.Compression.CompressionLevel]::Optimal) | Out-Null
        }
    } finally { $archive.Dispose() }
} finally { $stream.Dispose() }
Write-Output "Created $destination"

