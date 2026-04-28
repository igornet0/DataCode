#Requires -Version 5.1
<#
.SYNOPSIS
  Установка Datacode из распакованного бандла в Program Files и системный PATH.
  Запускайте PowerShell от имени администратора.
#>
$ErrorActionPreference = "Stop"

$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
  [Security.Principal.WindowsBuiltInRole]::Administrator
)
if (-not $isAdmin) {
  Write-Error "Запустите PowerShell от имени администратора (ПКМ → Запуск от имени администратора)."
  exit 1
}

$BundleDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BinDir = Join-Path $BundleDir "bin"
if (-not (Test-Path -LiteralPath $BinDir -PathType Container)) {
  $BinDir = $BundleDir
}

$Dest = Join-Path $env:ProgramFiles "Datacode"
$bins = @(
  @{ Name = "datacode";      File = "datacode.exe" },
  @{ Name = "dpm";          File = "dpm.exe" },
  @{ Name = "datacode-server"; File = "datacode-server.exe" }
)

$missing = @()
foreach ($b in $bins) {
  $p = Join-Path $BinDir $b.File
  if (-not (Test-Path -LiteralPath $p -PathType Leaf)) {
    $missing += $b.File
  }
}
if ($missing.Count -gt 0) {
  Write-Error "В папке бинарников не найдено: $($missing -join ', '). Ожидается каталог: $BinDir"
  exit 1
}

if (-not (Test-Path -LiteralPath $Dest)) {
  New-Item -ItemType Directory -Path $Dest | Out-Null
}

foreach ($b in $bins) {
  $src = Join-Path $BinDir $b.File
  $dst = Join-Path $Dest $b.File
  Copy-Item -LiteralPath $src -Destination $dst -Force
  Write-Host "Установлено: $dst"
}

$machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
$paths = if ($machinePath) { $machinePath -split ";" } else { @() }
$normDest = $Dest.TrimEnd("\")
$already = $paths | ForEach-Object { $_.TrimEnd("\") } | Where-Object { $_ -eq $normDest }
if (-not $already) {
  $newPath = if ($machinePath) { "$machinePath;$Dest" } else { $Dest }
  [Environment]::SetEnvironmentVariable("Path", $newPath, "Machine")
  $env:Path = "$env:Path;$Dest"
  Write-Host "В PATH (Machine) добавлено: $Dest"
} else {
  Write-Host "PATH уже содержит: $Dest"
}

Write-Host ""
Write-Host "Перезапустите терминал и проверьте: datacode --version"
