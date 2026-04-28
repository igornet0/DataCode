#Requires -Version 5.1
<#
.SYNOPSIS
  Удаляет DataCode из Program Files и убирает путь из системного PATH.
  Запуск: PowerShell от имени администратора.
#>
$ErrorActionPreference = "Stop"

$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
  [Security.Principal.WindowsBuiltInRole]::Administrator
)
if (-not $isAdmin) {
  Write-Error "Запустите PowerShell от имени администратора."
  exit 1
}

$Dest = Join-Path $env:ProgramFiles "Datacode"
$files = @("datacode.exe", "dpm.exe", "datacode-server.exe")

if (Test-Path -LiteralPath $Dest) {
  foreach ($f in $files) {
    $p = Join-Path $Dest $f
    if (Test-Path -LiteralPath $p) {
      Remove-Item -LiteralPath $p -Force
      Write-Host "Удалено: $p"
    }
  }
  if ((Get-ChildItem -LiteralPath $Dest -ErrorAction SilentlyContinue | Measure-Object).Count -eq 0) {
    Remove-Item -LiteralPath $Dest -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "Каталог удалён: $Dest"
  } else {
    Write-Host "Каталог оставлен (остались файлы): $Dest"
  }
} else {
  Write-Host "Каталог не найден: $Dest"
}

$normDest = $Dest.TrimEnd("\")
$machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
if ($machinePath) {
  $paths = $machinePath -split ";" | Where-Object { $_.TrimEnd("\") -ne $normDest -and $_ -ne "" }
  $newPath = $paths -join ";"
  [Environment]::SetEnvironmentVariable("Path", $newPath, "Machine")
  Write-Host "PATH (Machine) обновлён (удалён $Dest при наличии)."
} else {
  Write-Host "PATH (Machine) пуст или недоступен."
}

Write-Host "Готово. Перезапустите терминал."
