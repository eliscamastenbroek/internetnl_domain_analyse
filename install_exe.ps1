$TOOLS_DIR="\\cbsp.nl\productie\primair\WTE\Werk\WTE_Python\Python3.9"
$cmd="pip install . --no-build-isolation --prefix $TOOLS_DIR -U"
$cmd = $cmd -replace '\s+', ' '
Write-Output $cmd
Invoke-Expression $cmd


