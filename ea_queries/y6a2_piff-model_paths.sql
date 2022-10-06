select fai.path, fai.filename
from miscfile m, proctag t, file_archive_info fai 
where t.tag='Y6A2_PIFF_V2' 
    and t.pfw_attempt_id=m.pfw_attempt_id 
    and m.filetype='piff_model' 
    and m.filename=fai.filename
    and m.band != 'Y';
