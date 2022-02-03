import sys

from explore4FGL.explore4FGL import *
from astropy.table import Table
from astropy.io import fits


def main(data_path):
    # Get only the first data fits
    with fits.open(data_path) as hdu_list:
        lat_point_source_catalog = to_source_4fgl_data(Table(hdu_list[1].data))

    # Get the histogram for Variability_Index field
    lat_point_source_catalog.get_hist(colname='Variability_Index', title='distribution Variability Index', savefig=True,
                                      xlog=True, ylog=True, range=(0, 500), bins=200, histtype='step')
    # Get the bi-logarithmic plot for Variability_Index and Variability2_Index
    lat_point_source_catalog.get_plot("Variability_Index", "Variability2_Index", savefig=True, xlog=True, ylog=True)

    # Get the All-sources galactic map plot grouped by type of source
    lat_point_source_catalog.galactic_visualization_plot(coord_type='galactic',
                                                         title='All sources Galactic Map',
                                                         color='CLASS1')
    # Filter only pulsar sources
    pd_psr_pwn = lat_point_source_catalog.df.loc[lat_point_source_catalog.df['CLASS1'].str.match('(PSR)|(PWN)')]
    psr_pwn = Source4FGLData(pd_psr_pwn)

    # Get Some Galactic plot for pulsar sources
    psr_pwn.galactic_visualization_plot(coord_type='galactic',
                                        title='Galactic Map Pulsar and Pulsar with nebula',
                                        color='CLASS1')
    psr_pwn.galactic_visualization_plot(coord_type='galactic',
                                        title='Galactic Distribution of Spectrum type for Pulsar grouped by '
                                              'SpectrumType',
                                        color='SpectrumType')
    psr_pwn.galactic_visualization_plot(coord_type='galactic',
                                        title='Galactic Distribution of Spectrum type for Pulsar colored by '
                                              'Conf_95_SemiMajor',
                                        color='Conf_95_SemiMajor')

    # Attempt to classify unclassified sources
    lat_point_source_catalog.predict_classification(threshold=10)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(f"{os.path.dirname(os.path.abspath(__file__))}/data/{sys.argv[1]}")
    else:
        main(f"{os.path.dirname(os.path.abspath(__file__))}/data/gll_psc_v22.fit")
