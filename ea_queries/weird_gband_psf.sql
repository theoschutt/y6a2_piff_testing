select gai.path||'/'||c.filename||gai.compression 
from proctag ti, proctag tc, image i, miscfile c, file_archive_info fai, file_archive_info gai 
where ti.tag='Y6A1_FINALCUT' 
    and ti.pfw_attempt_id=i.pfw_attempt_id 
    and i.filename='D00372104_g_c01_r3539p01_immasked.fits'
    and tc.tag='Y6A2_PIFF_V3'
    and tc.pfw_attempt_id=c.pfw_attempt_id 
    and c.filetype='piff_model_stats' 
    and c.expnum=i.expnum 
    -- and c.ccdnum=i.ccdnum 
    and i.filename=fai.filename 
    and c.filename=gai.filename;
