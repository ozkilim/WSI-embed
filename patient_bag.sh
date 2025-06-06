# python CLAM/join_patient_bags.py \
#   --cohort_csv /mnt/ncshare/ozkilim/PANTHER/splits/OV-Survival/test_CPTAC.csv \
#   --base_path /tank/WSI_data/Ovarian_WSIs/CPTAC_OV/CLAM/20X/virchow_v2/pt_files \
#   --output_dir /tank/WSI_data/Ovarian_WSIs/CPTAC_OV/CLAM/20X/virchow_v2/pt_files_joined \
#   --overwrite



#   python CLAM/join_patient_bags.py \
#   --cohort_csv /mnt/ncshare/ozkilim/Ovarian_HRD_pred/OV_RNA/PANTHER/src/splits/classification/HGSOC_WSI_ALL/train.csv \
#   --base_path /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/20X/virchow_v2/pt_files \
#   --output_dir /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/20X/virchow_v2/pt_files_joined \
#   --overwrite


# HGSOC ...

# python CLAM/join_patient_bags.py \
# --cohort_csv /mnt/ncshare/ozkilim/Ovarian_HRD_pred/response_labels_inspection/response_cohorts/PTRC_response_slides_included.csv \
# --base_path /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/20X/uni_v1/pt_files \
# --output_dir /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/20X/uni_v1/pt_files_joined \
# --overwrite
  



# python CLAM/join_patient_bags.py \
# --cohort_csv /mnt/ncshare/ozkilim/Ovarian_HRD_pred/response_labels_inspection/response_cohorts/PTRC_response_slides_included.csv \
# --base_path /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/20X/virchow_v2/pt_files \
# --output_dir /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/20X/virchow_v2/pt_files_joined \
# --overwrite
  


# python CLAM/join_patient_bags.py \
# --cohort_csv /mnt/ncshare/ozkilim/Ovarian_HRD_pred/response_labels_inspection/response_cohorts/PTRC_response_slides_included.csv \
# --base_path /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/20X/H-optimus-0/pt_files \
# --output_dir /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/20X/H-optimus-0/pt_files_joined \
# --overwrite
  

### Need to run for CPTAC ...


python CLAM/join_patient_bags.py \
--cohort_csv /mnt/ncshare/ozkilim/Ovarian_HRD_pred/response_labels_inspection/response_cohorts/ovar_image_data_mar22_d032222.csv \
--base_path /tank/WSI_data/Ovarian_WSIs/PLCO-OV/CLAM/20X/20X/uni_v1/pt_files \
--output_dir /tank/WSI_data/Ovarian_WSIs/PLCO-OV/CLAM/20X/20X/uni_v1/pt_files_joined \
--overwrite
  
# Chnge df or make copy befroe with new col names... etc etc... then rerun preds here... to see if diff... 

