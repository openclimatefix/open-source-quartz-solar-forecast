import unittest

# What we are testing
from vrm import VRM_API


class TestAPIMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Settings that will be kept throughout all test
        """

        pass

    def setUp(self):
        """
        Settings that will be applied to every test
        """
        self.demo_user_id = 22
        self.demo_site_id = 2286
        self.client = VRM_API(demo=True)
        pass

    def tearDown(self):
        pass

    def test_api_init(self):
        """
        Testing that login for api is working
        """
        self.assertEqual(self.client.is_initialized(), True)

    def test_api_fetch(self):
        """
        Testing getting counters for the last week of a site
        """
        now = "1466936333"
        last_week_now = "1466331533"
        result = self.client.get_counters_site(self.demo_site_id, last_week_now, now)
        self.assertEqual("records" in result, True)

    def test_consumption_stats(self):
        """
        Test the returning of data for the installations
        stats
        """
        result = self.client.get_consumption_stats(self.demo_site_id)
        self.assertTrue(result["success"])
        self.assertTrue("records" in result)

    def test_kwh_stats(self):
        """
        Test the returning of the kwh stats
        """
        result = self.client.get_consumption_stats(inst_id=self.demo_site_id)
        self.assertTrue(result["success"])
        self.assertTrue("records" in result)

    def test_consumption_aggr_stats(self):
        """
        Test the aggregated stats
        """
        result = self.client.consumption_aggr_stats(inst_id=self.demo_site_id)
        self.assertTrue(result["success"])
        self.assertTrue("records" in result)

    def test_kwh_aggr_stats(self):
        """
        Test the kwh aggregated stats
        """
        result = self.client.kwh_aggr_stats(inst_id=self.demo_site_id)
        self.assertTrue(result["success"])
        self.assertTrue("records" in result)

    def test_graph_widgets(self):
        """
        Testing the graph widgets
        """
        result = self.client.graph_widgets(self.demo_site_id, ["IV1", "IV2"])
        self.assertTrue(result["success"])
        self.assertTrue("records" in result)

    def test_user_sites(self):
        """
        Test fetching of sites with reports enabled
        """
        # Testing the user sites simplified version
        sites_normal = self.client.get_user_sites(self.demo_user_id)
        self.assertTrue(sites_normal["success"])
        self.assertTrue("records" in sites_normal)

        # Testing the users sites extended version
        sites_normal_extended = self.client.get_user_sites(
            self.demo_user_id, extended=True
        )
        self.assertTrue(sites_normal_extended["success"])
        self.assertTrue("records" in sites_normal_extended)
        self.assertTrue("tags" in sites_normal_extended["records"][0])

        self.assertEqual(
            len(sites_normal["records"]), 25
        )  # Warning might break if user permissions change


if __name__ == "__main__":
    unittest.main()
