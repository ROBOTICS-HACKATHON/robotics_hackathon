# screencap.ps1 - saves a screenshot of this computer's screen about 4 times a second,
# so the robot (./lr with CAMERA=screen) can "see" the CAPTCHA page without a camera.
#
# Start it yourself in a separate PowerShell window and leave it running:
#   powershell -ExecutionPolicy Bypass -File "$HOME\Downloads\screencap.ps1"
# Stop it with Ctrl+C. It only writes one picture file (overwritten each time); nothing else.

param(
    # where the robot looks for the picture: the outputs folder of the repo inside WSL
    [string]$Out = "\\wsl.localhost\Ubuntu-24.04\home\mehras\robotics_hackathon\outputs\screen_live.jpg",
    [int]$EveryMs = 250
)

# Real pixels on high-DPI screens (otherwise Windows hands back a scaled-down, cropped picture)
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class DpiAware { [DllImport("user32.dll")] public static extern bool SetProcessDPIAware(); }
"@
[DpiAware]::SetProcessDPIAware() | Out-Null
Add-Type -AssemblyName System.Windows.Forms, System.Drawing

$tmp = "$Out.tmp.jpg"
$jpeg = [System.Drawing.Imaging.ImageCodecInfo]::GetImageEncoders() | Where-Object { $_.MimeType -eq "image/jpeg" }
$quality = New-Object System.Drawing.Imaging.EncoderParameters 1
$quality.Param[0] = New-Object System.Drawing.Imaging.EncoderParameter([System.Drawing.Imaging.Encoder]::Quality, [long]92)

Write-Host "Saving a screenshot every $EveryMs ms to $Out"
Write-Host "Leave this window open while the robot runs. Ctrl+C to stop."
$n = 0
while ($true) {
    $area = [System.Windows.Forms.SystemInformation]::VirtualScreen   # all monitors
    $bmp = New-Object System.Drawing.Bitmap $area.Width, $area.Height
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.CopyFromScreen($area.Left, $area.Top, 0, 0, $bmp.Size)
    $bmp.Save($tmp, $jpeg, $quality)
    $g.Dispose(); $bmp.Dispose()
    Move-Item -Force $tmp $Out          # swap in the finished picture (never half-written)
    $n++
    if ($n % 40 -eq 0) { Write-Host "  $n screenshots ($($area.Width)x$($area.Height))" }
    Start-Sleep -Milliseconds $EveryMs
}
