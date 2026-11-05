param(
    [Parameter(Mandatory=$true)][string]$Vectors,
    [string]$Data = '../bgc_prepared',
    [ValidateSet('All','Train','Prepare','Queries','Centroid','OTPrepare','Sinkhorn')]
    [string]$Stage = 'All',
    [ValidateRange(1,1000000)][int]$Terms = 1000,
    [double]$Reg = 0.05,
    [string]$Device = 'cuda:0',
    [int[]]$K1 = @(3,8,8),
    [int[]]$K2 = @(2,4,4),
    [ValidateRange(0,1000000)][int]$Limit = 0
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if ($K1.Count -ne 3 -or $K2.Count -ne 3) { throw 'K1 and K2 require three values.' }
for ($i = 0; $i -lt 3; $i++) {
    if ($K1[$i] -lt 1 -or $K2[$i] -lt 1 -or $K2[$i] -gt $K1[$i]) {
        throw 'K values must be positive and K2 cannot exceed K1.'
    }
}
if ($Reg -le 0) { throw 'Reg must be positive.' }
if (-not (Test-Path $Vectors -PathType Leaf)) { throw "Missing vectors: $Vectors" }
if (-not (Test-Path "$Data/train.jsonl" -PathType Leaf)) { throw "Missing prepared data: $Data" }

function Invoke-Python {
    param([string[]]$PythonArgs)
    & python @PythonArgs
    if ($LASTEXITCODE -ne 0) { throw "Python failed: $($PythonArgs -join ' ')" }
}

$suffix = if ($Limit -gt 0) { "first$Limit" } else { 'all' }
$queryFile = "./queries_bgc_dev_$suffix.jsonl"
$distribution = "./ot_bgc_terms$Terms.sqlite"
$tag = "k1_$($K1 -join '-')_k2_$($K2 -join '-')_$suffix"
$regString = $Reg.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$regTag = $regString.Replace('.', 'p')

if ($Stage -in @('All','Train')) {
    Invoke-Python -PythonArgs @('bgc_workflow.py','train','--data',$Data,'--model','./bgc_model')
}
if ($Stage -in @('All','Prepare')) {
    Invoke-Python -PythonArgs @('bgc_workflow.py','prepare','--model','./bgc_model',
        '--store','./nodes_bgc.sqlite','--vectors',$Vectors)
}
if ($Stage -in @('All','Queries')) {
    $queryArgs = @('bgc_workflow.py','queries','--model','./bgc_model',
        '--input',"$Data/dev.jsonl",'--output',$queryFile)
    if ($Limit -gt 0) { $queryArgs += @('--limit',"$Limit") }
    Invoke-Python -PythonArgs $queryArgs
}
$common = @('pipelines_bgc_gpu.py','worker','--store','./nodes_bgc.sqlite',
    '--queries',$queryFile,'--training','.','--vectors',$Vectors,
    '--vector-format','fasttext_vec','--topk1-per-level') + @($K1 | ForEach-Object { "$_" }) +
    @('--topk2-per-level') + @($K2 | ForEach-Object { "$_" }) +
    @('--lookahead','--gate','gold','--weights','0.33','0.33','0.33')
if ($Stage -in @('All','Centroid')) {
    Invoke-Python -PythonArgs ($common + @('--method','centroid','--device','cpu',
        '--output',"./result_bgc_dev_centroid_$tag"))
}
if ($Stage -in @('All','OTPrepare')) {
    Invoke-Python -PythonArgs @('prepare_ot_distributions.py','--pipeline','./pipelines_bgc_gpu.py',
        '--store','./nodes_bgc.sqlite','--training','.','--vectors',$Vectors,
        '--vector-format','fasttext_vec','--ot-max-terms',"$Terms",'--output',$distribution)
}
if ($Stage -in @('All','Sinkhorn')) {
    $cells = [long]$Terms * [long]$Terms
    Invoke-Python -PythonArgs ($common + @('--method','centroid_sinkhorn','--device',$Device,
        '--precomputed-ot',$distribution,'--ot-dtype','float64','--sinkhorn-solver','sinkhorn',
        '--ground-cost','cosine','--reg',$regString,'--ot-max-terms',"$Terms",'--max-cells',"$cells",
        '--profile-ot','--output',"./result_bgc_dev_sinkhorn_terms${Terms}_reg${regTag}_$tag"))
}
Write-Host "Completed stage: $Stage"
