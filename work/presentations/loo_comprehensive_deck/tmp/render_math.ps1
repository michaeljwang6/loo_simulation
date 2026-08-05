$ErrorActionPreference = 'Stop'
$mathDir = 'C:\Users\A8327\OneDrive\Documents\OI\work\presentations\loo_comprehensive_deck\tmp\math'
$edge = 'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'

Get-ChildItem -LiteralPath $mathDir -Filter '*.html' | ForEach-Object {
    $png = Join-Path $mathDir ($_.BaseName + '.png')
    $uri = 'file:///' + ($_.FullName -replace '\\', '/')
    & $edge '--headless=new' '--disable-gpu' '--disable-gpu-compositing' '--disable-software-rasterizer' '--hide-scrollbars' '--default-background-color=00000000' '--window-size=2400,300' ("--screenshot=$png") $uri
    for ($attempt = 0; $attempt -lt 50 -and -not (Test-Path -LiteralPath $png); $attempt++) {
        Start-Sleep -Milliseconds 200
    }
    if (-not (Test-Path -LiteralPath $png)) {
        throw "Equation render failed: $($_.BaseName)"
    }
}
