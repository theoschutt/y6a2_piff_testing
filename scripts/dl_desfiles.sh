filename="~/gband_proj/ea_queries/piffimg_allrows_1_sorted.csv"
urlprepend="https://desar2.cosmology.illinois.edu/DESFiles/desarchive/"

# get file paths from csvs for images, bkg images, and star catalogs
mapfile -s 1 -n 10 -t file_list < <( awk -F "\"*,\"*" '{print $4"/"$5}' $filename | sed 's/b\x27//g; s/\x27//g' )

for i in ${!cat_list[@]}; do
    # get file paths and names
    im="$urlprepend${cat_list[$i]}"
    
