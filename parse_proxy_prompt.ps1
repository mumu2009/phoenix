param([string]$LogPath='proxy_prompt.log')
$lines = Get-Content $LogPath -Tail 80
for ($i = $lines.Count - 1; $i -ge 0; $i--) {
    if ($lines[$i] -match 'PROMPT=') {
        $idx = $lines[$i].IndexOf('PROMPT=')
        $prompt = $lines[$i].Substring($idx + 7)
        $i++
        while ($i -lt $lines.Count -and $lines[$i] -notmatch '^\[STATUS=') {
            $prompt += "`n" + $lines[$i]
            $i++
        }
        if ($prompt.Length -gt 500) {
            $prompt.Substring(0, 500)
        } else {
            $prompt
        }
        break
    }
}
