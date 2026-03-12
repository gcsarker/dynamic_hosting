$bucket="oedi-data-lake"

$prefix="nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2022/resstock_tmy3_release_1.1/timeseries_individual_buildings/by_state/upgrade=0/state=NY/"

$files = aws s3api list-objects-v2 `
--bucket $bucket `
--prefix $prefix `
--no-sign-request `
--query "Contents[].Key" `
--output text

$offset = 100
$n = 200

$subset = $files.Split("`t")[$offset..($offset+$n-1)]

foreach ($f in $subset) {
    aws s3 cp --no-sign-request `
    "s3://$bucket/$f" `
    ".\dataset\$(Split-Path $f -Leaf)"
}