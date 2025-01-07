python CLAM/join_patient_bags.py \
  --cohort_csv /mnt/ncshare/ozkilim/PANTHER/splits/OV-Survival/test_CPTAC.csv \
  --base_path /tank/WSI_data/Ovarian_WSIs/CPTAC_OV/CLAM/20X/virchow_v2/pt_files \
  --output_dir /tank/WSI_data/Ovarian_WSIs/CPTAC_OV/CLAM/20X/virchow_v2/pt_files_joined \
  --overwrite



  python CLAM/join_patient_bags.py \
  --cohort_csv /mnt/ncshare/ozkilim/Ovarian_HRD_pred/OV_RNA/PANTHER/src/splits/classification/HGSOC_WSI_ALL/train.csv \
  --base_path /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/20X/virchow_v2/pt_files \
  --output_dir /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/20X/virchow_v2/pt_files_joined \
  --overwrite



  python CLAM/join_patient_bags.py \
  --cohort_csv /mnt/ncshare/ozkilim/MIL/data/OV_cohorts/test_MSK.csv \
  --base_path /tank/WSI_data/Ovarian_WSIs/MSK-IMPACT/CLAM/20X/virchow_v2/pt_files \
  --output_dir /tank/WSI_data/Ovarian_WSIs/MSK-IMPACT/CLAM/20X/virchow_v2/pt_files_joined \
  --overwrite
  


