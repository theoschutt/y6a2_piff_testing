with open('y6a2_mdet_run1_se_ims_in_coadds.txt') as f:
    lines = f.readlines()

with open('y6a2_mdet_run1_piffmods_in_coadds.csv', 'w') as p:
    p.write('FILENAME\n')
    for line in lines:
        pline = line.strip('\n')
        pline = pline.strip('red/.fz')
        pline = pline.replace('immasked', 'piff-model')
        p.write(pline + '\n')
