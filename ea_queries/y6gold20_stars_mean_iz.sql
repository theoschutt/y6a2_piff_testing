select avg(y.psf_mag_aper_8_i-y.psf_mag_aper_8_z),median(y.psf_mag_aper_8_i-y.psf_mag_aper_8_z)
from y6_gold_2_0 y
where y.ext_mash=0
    and (y.psf_mag_aper_8_i < 22 or y.psf_mag_aper_8_z < 22.)
    and (y.psf_flux_flags_i =0 and y.psf_flux_flags_z =0);