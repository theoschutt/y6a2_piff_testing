select avg(y.psf_mag_aper_8_g-y.psf_mag_aper_8_i),median(y.psf_mag_aper_8_g-y.psf_mag_aper_8_i)
from y6_gold_2_0 y
where y.ext_mash=0
    and (y.psf_mag_aper_8_g < 22 or y.psf_mag_aper_8_i < 22.)
    and (y.psf_flux_flags_g =0 and y.psf_flux_flags_i =0)
    and exists (select 1 from gaia_dr2_x_y6_gold_2_0 x where x.coadd_object_id=y.coadd_object_id);