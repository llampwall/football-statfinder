param(
  [Parameter(Mandatory = $false)]
  [string]$Message = "Codex completed the requested task."
)

if ([string]::IsNullOrWhiteSpace($Message)) {
  $Message = "Codex completed the requested task."
}

[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$payload = @{ content = $Message.Trim() } | ConvertTo-Json -Compress

Invoke-RestMethod `
  -Uri "https://discord.com/api/webhooks/1427952554234478602/tJ-0afLyyjL-uPalbWLpbLygUFQahInTnvI6gBRUaWAqdKgy7evAZUkxmDW__cM1Zxzh" `
  -Method POST `
  -ContentType "application/json; charset=utf-8" `
  -Body $payload
