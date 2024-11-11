./embed.sh configs/ovarian/MSKCC_20X.yaml &
./embed.sh configs/ovarian/zoltan_BRCA_blind_20X.yaml &
./embed.sh configs/ovarian/TCGA-OV_20X.yaml &
wait
./embed.sh configs/ovarian/CPTAC-OV_20X.yaml 
./embed.sh configs/ovarian/HGSOC_20X.yaml 