import unittest

from astropy.io import fits
from astropy.table import Table

from explore4FGL.explore4FGL import *


class TestUtilsMethods(unittest.TestCase):

    def test_save_or_show_plot(self):
        """
        Check if the method save the plot image in the right output folder
        """
        x = range(10)
        y = range(5, 15)
        plt.figure()
        plt.hist(x, y)
        test_plot_name = "test_plot"
        save_or_show_plot(True, test_plot_name)
        output_file = f"{output_path}/{test_plot_name}.png"
        self.assertTrue(os.path.exists(output_file))
        os.remove(output_file)


source_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = f"{source_path}/data/gll_psc_v22.fit"

try:  # trying to get LAT_Point_Source_Catalog and ExtendedSources from the fits file
    with fits.open(data_path) as hdu_list:
        test_df = to_source_4fgl_data(Table(hdu_list[1].data))
except Exception as e:
    print(e)
    sys.exit(1)


class TestSource4FGLData(unittest.TestCase):

    def test_from_astropy_table(self):
        """
        Check if the method correctly convert an astropy table in a Source4FGLData object
        """
        test_source_4fgl = Source4FGLData.from_4fgl_file(data_path)
        self.assertIsInstance(test_source_4fgl, Source4FGLData)

    def test_get_hist(self):
        """
        Check if the method creates the histogram
        """
        test_histogram_name = 'test histogram'
        test_df.get_hist(colname='RAJ2000',
                         title=test_histogram_name,
                         savefig=True)
        output_file = f"{output_path}/{test_histogram_name}.png"
        self.assertTrue(os.path.exists(output_file))
        os.remove(output_file)

    def test_get_plot(self):
        """
        Check if the method creates the plot
        """
        test_plot_name = "test plot"
        test_df.get_plot(col_x='RAJ2000',
                         col_y='DEJ2000',
                         title='test plot',
                         savefig=True)
        output_file = f"{output_path}/{test_plot_name}.png"
        self.assertTrue(os.path.exists(output_file))
        os.remove(output_file)

    def test_galactic_visualization_plot(self):
        """
        Check if the method creates the galactic plot
        """
        test_galactic_plot_name = "test galactic plot"
        test_df.galactic_visualization_plot(coord_type='galactic', title=test_galactic_plot_name, savefig=True)
        output_file_galactic = f"{output_path}/{test_galactic_plot_name}.png"
        test_equatorial_plot_name = "test equatorial plot"
        test_df.galactic_visualization_plot(title=test_equatorial_plot_name, savefig=True)
        output_file_equatorial = f"{output_path}/{test_equatorial_plot_name}.png"
        self.assertTrue(os.path.exists(output_file_galactic))
        self.assertTrue(os.path.exists(output_file_equatorial))
        os.remove(output_file_galactic)
        os.remove(output_file_equatorial)

    def test_classification_predictor(self):
        """
        Check if the method creates the accuracy plot
        """
        test_df.predict_classification()
        output_file = f"{output_path}/DT_Accuracy.png"
        self.assertTrue(os.path.exists(output_file))
        os.remove(output_file)


if __name__ == '__main__':
    unittest.main()
