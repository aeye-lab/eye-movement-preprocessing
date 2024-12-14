batch="$1"
Rscript --vanilla baseline_analyses.R -r FPRT -i 3000 -b ${batch} > logs/${batch}_FPRT.log &
Rscript --vanilla baseline_analyses.R -r TFT -i 3000 -b ${batch} > logs/${batch}_TFT.log &
Rscript --vanilla baseline_analyses.R -r Fix -i 3000 -b ${batch} > logs/${batch}_Fix.log &
Rscript --vanilla baseline_analyses.R -r FPReg -i 3000 -b ${batch} > logs/${batch}_FPReg.log &
