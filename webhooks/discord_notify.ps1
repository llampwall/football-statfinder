param(
  [string]$Message = "✅ Codex task completed."
)

# Force TLS 1.2 for older PowerShells
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# Build JSON *string* (not a hashtable)
$payload = @{ content = $Message } | ConvertTo-Json -Compress

# Optional sanity check: should print "String"
# ($payload | Get-Member).TypeName

Invoke-RestMethod `
  -Uri "https://discord.com/api/webhooks/1427952554234478602/tJ-0afLyyjL-uPalbWLpbLygUFQahInTnvI6gBRUaWAqdKgy7evAZUkxmDW__cM1Zxzh"`
  -Method POST `
  -ContentType "application/json; charset=utf-8" `
  -Body $payload