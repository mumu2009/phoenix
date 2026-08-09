param([string]$baseUrl = "http://127.0.0.1:5081")
$ErrorActionPreference = "Stop"

Write-Host "[INFO] generating smoke media..."
$genPy = @"
import base64, io, math, os, struct, wave
from PIL import Image

img = Image.new('RGB', (224, 224))
for y in range(224):
    for x in range(224):
        img.putpixel((x, y), (x * 255 // 224, y * 255 // 224, 128))
img.save('build/tmp/smoke_image.png')

with wave.open('build/tmp/smoke_16000.wav', 'w') as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(16000)
    samples = bytearray()
    for i in range(16000):
        v = int(32767 * math.sin(2 * math.pi * 440 * i / 16000))
        samples.extend(struct.pack('<h', v))
    f.writeframes(bytes(samples))
"@
$genPy | Out-File -Encoding utf8 "build\tmp\gen_smoke.py"
& ".\Python314\python.exe" "build\tmp\gen_smoke.py"
$imgPath = "D:\_phoenix\_079\v6.0Alixander\phoenix\build\tmp\smoke_image.png"
$wavPath = "D:\_phoenix\_079\v6.0Alixander\phoenix\build\tmp\smoke_16000.wav"
if (-not (Test-Path $imgPath)) { throw "failed to generate image" }
if (-not (Test-Path $wavPath)) { throw "failed to generate audio" }
$wavB64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes($wavPath))

Write-Host "[INFO] /vision/analyze (imagePath)..."
$vision = Invoke-RestMethod -Method POST -Uri "$baseUrl/vision/analyze" -ContentType 'application/json' -Body (ConvertTo-Json -Compress @{imagePath = 'build/tmp/smoke_image.png'; sessionId = 'aa-smoke'})
if (-not $vision.ok) { throw "vision/analyze not ok: $($vision | ConvertTo-Json)" }
if ($vision.embeddingDim -ne 128) { throw "vision embeddingDim unexpected: $($vision.embeddingDim)" }
Write-Host "  ok=$($vision.ok) backend=$($vision.backend) embeddingDim=$($vision.embeddingDim)"

Write-Host "[INFO] /speech/analyze..."
$speech = Invoke-RestMethod -Method POST -Uri "$baseUrl/speech/analyze" -ContentType 'application/json' -Body (ConvertTo-Json -Compress @{audioBase64 = $wavB64; sessionId = 'aa-smoke'})
if (-not $speech) { throw "speech/analyze returned empty" }
Write-Host "  response keys: $($speech | Get-Member -MemberType NoteProperty | ForEach-Object { $_.Name })"

Write-Host "[PASS] multimodal smoke"
exit 0
