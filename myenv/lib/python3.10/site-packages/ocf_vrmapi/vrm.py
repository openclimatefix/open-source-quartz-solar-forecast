# Main imports
import datetime
import logging

# for testing
from datetime import datetime

import requests

# setup Logger
logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
logger.addHandler(ch)


class VRM_API:
    # Get this
    RETRY = 3

    def __init__(self, username=None, password=None, demo=False):
        """
        Initialise API for Victron VRM
        @param - username
        @param - password
        @param - config
        """

        self._initialized = False
        self.API_ENDPOINT = "https://vrmapi.victronenergy.com"

        self._auth_token = ""
        self._ses = requests.Session()
        self.user_id = ""

        self.DEMO_AUTH_ENDPOINT = self.API_ENDPOINT + "/v2/auth/loginAsDemo"
        self.AUTH_ENDPOINT = self.API_ENDPOINT + "/v2/auth/login"
        self.QUERY_ENDPOINT = self.API_ENDPOINT + "/v2/installations/{inst_id}/stats"
        self.AGGR_STATS_ENDPOINT = (
            self.API_ENDPOINT + "/v2/installations/{inst_id}/overallstats"
        )
        self.USER_ENDPOINT = self.API_ENDPOINT + "/v2/admin/users"
        self.USER_SITE_ENDPOINT = (
            self.API_ENDPOINT + "/v2/users/{user_id}/installations"
        )
        self.WIDGETS_ENDPOINT = (
            self.API_ENDPOINT + "/v2/installations/{inst_id}/widgets/{widget_type}"
        )
        self.DIAG_ENDPOINT = (
            self.API_ENDPOINT + "/v2/installations/{inst_id}/diagnostics"
        )

        if demo:  # Login as demo else with credentials
            self._initialized = self._login_as_demo()
        else:
            if username and password:
                self.username = username
                self.password = password
            else:
                raise Exception("No username or password provided")

            logger.debug("Initializing API with username %s " % (self.username))
            self._initialized = self._login()

    def initialize(self):
        """
        Login and get auth token
        """
        self._initialized = self._login()

    def get_counters_site(self, site_id, start, end, query_interval="days"):
        """
        Get counters for a given site
        @param - site_id
        @param - start
        @param - end

        """

        result = self._prepare_query_request(
            site_id, start, end, query_interval=query_interval
        )

        logger.debug("Result for query %s" % result)

        # Make format nice
        return result

    def is_initialized(self):
        """
        Return the status of the API
        """
        return self._initialized

    def _is_initialized(self):
        """
        Internal helper function to check if api is initialized
        """
        if not self._initialized:
            logger.error("API not initialized")
            return False
        return True

    def get_user_sites(self, user_id, extended=False):
        """
        Download list of sites for logged in user
        @param - user_id
        @param - extended ( boolean value for extra site info)
        """
        if not self._is_initialized():
            return None

        request_url = self.USER_SITE_ENDPOINT.format(user_id=user_id)

        if not extended:
            sites = self._send_query_request(request_url)
        else:
            sites = self._send_query_request(request_url, data_dict={"extended": "1"})
        logger.debug("got sites for user %s %s" % (user_id, sites))
        return sites

    def get_user_sites_reporting(self, user_id):
        """
        Download list of sites for logged in user
        @param - user_id
        """
        if not self._is_initialized():
            return None

        request_url = self.USER_SITE_ENDPOINT.format(user_id=user_id)
        site = self._send_query_request(request_url)
        if site.has_key("records"):
            site["records"] = filter(lambda x: x["reports_enabled"], site["records"])
            logger.debug(
                "got site for user with reporting enabled %s %s" % (user_id, site)
            )
            return site
        return {}

    def get_all_users(self):
        """
        Get a list of all users registered
        """
        if not self._is_initialized():
            return None
        meta = {"count": 99999}
        logging.debug("Fetching users")
        users = self._send_query_request(self.USER_ENDPOINT, data_dict=meta)
        return users

    def get_consumption_stats(self, inst_id, start=None, end=None):
        """
        Returns the consumptions statistics for a given site
        @params - inst_id (installation id)
        @params - start ( A python datetime to start from)
        @params - end (A python datetime to stop to)
        """
        if not self._is_initialized():
            return None

        if start and end:
            data_dict = {
                "type": "consumption",
                "start": datetime.datetime(start).timestamp(),
                "end": datetime.datetime(end).timestamp(),
            }
        else:
            data_dict = {
                "type": "consumption",
            }

        request_url = self.QUERY_ENDPOINT.format(inst_id=inst_id)
        stats = self._send_query_request(request_url, data_dict=data_dict)
        logger.debug("The stats consumption got from the api endpoint is %s " % stats)
        return stats

    def get_diag(self, inst_id, data_points=100):
        """
        @params - inst_id (installation id)
        """
        if not self._is_initialized():
            return None

        data_dict = {"count": data_points}
        request_url = self.DIAG_ENDPOINT.format(inst_id=inst_id)
        diag_request = self._send_query_request(request_url, data_dict)
        return diag_request

    def get_kwh_stats(self, inst_id, start: datetime = None, end: datetime = None):
        """
        Returns the kwhs statistics for a given site
        @params - inst_id (installation id)
        @params - start ( A python datetime to start from)
        @params - end ( A python datetime to stop to)
        """
        if not self._is_initialized():
            return None

        if start and end:
            data_dict = {
                "type": "kwh",
                "start": start.timestamp(),
                "end": end.timestamp(),
                "interval": "15mins",
            }
        else:
            data_dict = {
                "type": "kwh",
            }
        request_url = self.QUERY_ENDPOINT.format(inst_id=inst_id)
        stats = self._send_query_request(request_url, data_dict)
        logger.debug("The kwh stats got from the api endpoint are %s" % stats)
        return stats

    def consumption_aggr_stats(self, inst_id):
        """
        Returns consumption aggreagated stats
        @params inst_id ( site id )
        """
        if not self._is_initialized():
            return None

        data_dict = {"type": "consumption"}
        request_url = self.AGGR_STATS_ENDPOINT.format(inst_id=inst_id)
        stats = self._send_query_request(request_url, data_dict)
        return stats

    def kwh_aggr_stats(self, inst_id):
        """
        Returns kwh aggregated statistics
        @params inst_id ( installation id )
        """
        if not self._is_initialized():
            return None

        data_dict = {"type": "kwh"}
        request_url = self.AGGR_STATS_ENDPOINT.format(inst_id=inst_id)
        stats = self._send_query_request(request_url, data_dict)
        return stats

    def graph_widgets(
        self, inst_id, measurement_codes, instance=None, start=None, end=None
    ):
        """
        Returns graph widgets for given measurements codes
        @param - inst_id (installation id)
        @param - measurement_codes (A List of the measurent codes)
        """
        if type(measurement_codes) is not list:
            raise Exception("The measurement codes should be an array")

        if not self._is_initialized():
            return None

        data_dict = {"attributeCodes[]": measurement_codes}

        if instance:
            data_dict["instance"] = instance

        if start and end:
            data_dict["start"] = datetime.datetime(start).timestamp()
            data_dict["end"] = datetime.datetime(end).timestamp()

        request_url = self.WIDGETS_ENDPOINT.format(inst_id=inst_id, widget_type="Graph")
        widgets = self._send_query_request(request_url, data_dict)
        return widgets

    def ve_bus_state_widget(self, inst_id, instance=None, start=None, end=None):
        """
        Returns the ve bus state widget
        @param - inst_id
        @param - instance
        @param - start
        @param - end
        """
        return self._state_graph_widgets(inst_id, "VeBusState", instance, start, end)

    def mppt_state_widget(self, inst_id, instance=None, start=None, end=None):
        """
        Returns the mppt state widget
        @param - inst_id
        @param - instance
        @param - start
        @parma - end
        """
        return self._state_graph_widgets(inst_id, "MPPTState", instance, start, end)

    def ve_bus_warning_and_alarms_wigdet(
        self, inst_id, instance=None, start=None, end=None
    ):
        """
        Returns teh ve bus warning and allarms widget
        @param - inst_id
        @param - instance
        @param - start
        @parma - end
        """
        return self._state_graph_widgets(
            inst_id, "VeBusWarningsAndAlarms", instance, start, end
        )

    def battery_summary_widget(self, inst_id, instance=None):
        """
        Returns the battery summary widget
        @param - inst_id
        @param - instance
        """
        return self._state_graph_widgets(inst_id, "BatterySummary", instance)

    def bms_diagnostics_widget(self, inst_id, instance=None):
        """
        Returns the bms diagnostic widget
        @param - inst_id
        @param - instance
        """
        return self._state_graph_widgets(inst_id, "BMSDiagnostics", instance)

    def historic_data_widget(self, inst_id, instance=None):
        """
        Returns historical data widget
        @param - inst_id
        @param - instance
        """
        return self._state_graph_widgets(inst_id, "HistoricData", instance)

    def io_extender_in_out_widget(self, inst_id, instance=None):
        """
        Returns io extender in out
        @param - inst_id
        @param - instance
        """
        return self._state_graph_widgets(inst_id, "IOExtenderInOut", instance)

    def lithium_bms_widget(self, inst_id, instance=None):
        """
        Returns lithium bms widget
        @param - inst_id
        @param - instance
        """
        return self._state_graph_widgets(inst_id, "LithiumBMS", instance)

    def motor_summary_widget(self, inst_id, instance=None):
        """
        Returns motor summary in out
        @param - inst_id
        @param - instance
        """
        return self._state_graph_widgets(inst_id, "MotorSummary", instance)

    def pv_inverter_status_widget(self, inst_id, instance=None):
        """
        Returns pv inverter status in out
        @param - inst_id
        @param - instance
        """
        return self._state_graph_widgets(inst_id, "PVInverterStatus", instance)

    def solar_charger_summary_widget(self, inst_id, instance=None):
        """
        Returns Solar Charger summary in out
        @param - inst_id
        @param - instance
        """
        return self._state_graph_widgets(inst_id, "SolarChargerSummary", instance)

    def status_widget(self, inst_id, instance=None):
        """
        Returns motor summary in out
        @param - inst_id
        @param - instance
        """
        return self._state_graph_widgets(inst_id, "Status", instance)

    def alarm_widget(self, inst_id):
        """
        Returns the alarm widget
        @param - inst_id
        """
        return self._state_graph_widgets(inst_id, "Alarm")

    def gps_widget(self, inst_id):
        """
        Returns the gps widget
        @param - inst_id
        """
        return self._state_graph_widgets(inst_id, "GPS")

    def hours_of_ac_widget(self, inst_id):
        """
        Returns hours of ac widget
        @param - inst_id
        """
        return self._state_graph_widgets(inst_id, "HoursOfAC")

    def _state_graph_widgets(
        self, inst_id, widget_name, instance=None, start=None, end=None
    ):
        """
        Internal function to make calls for state widget functions
        @param - inst_id
        @param - widget_name
        @param - instance
        @param - start ( Python datetime object)
        @param - end ( Python datetime object)
        """
        if not self._is_initialized():
            return None

        data_dict = {}

        if start and end:
            data_dict["start"] = datetime.datetime(start).timestamp()
            data_dict["end"] = datetime.datetime(start).timestamp()

        if instance:
            data_dict["instance"] = instance

        request_url = self.WIDGETS_ENDPOINT.format(
            inst_id=inst_id, widget_type=widget_name
        )
        widgets = self._send_query_request(request_url, data_dict)
        return widgets

    def _login(self):
        """
        Login to API and get token
        """
        data_packet = {"username": self.username, "password": self.password}

        result = requests.post(self.AUTH_ENDPOINT, json=data_packet)

        if result.status_code == 200:
            response_json = result.json()
            self._auth_token = response_json["token"]
            self.user_id = response_json["idUser"]
            logger.debug("API initialized with token %s" % (self._auth_token))
            return True
        elif result.status_code == 401:
            logger.error("Unable to authenticate")
            return False
        else:
            logger.error(
                "Problem authenticating status code:%s  text:%s"
                % (result.status_code, result.text)
            )
            return False

    def _login_as_demo(self):
        """
        Login using the api demo,
        used for testing
        """
        result = requests.get(self.DEMO_AUTH_ENDPOINT)

        if result.status_code == 200:
            response_json = result.json()
            self._auth_token = response_json["token"]
            logger.debug(
                "API initialized with demo account , token: %s" % (self._auth_token)
            )
            return True
        else:
            logger.error("Unable to login as demo")
            return False

    def _prepare_query_request(
        self, site_id, start_epoch, end_epoch, query_interval, query_type="kwh"
    ):
        """
        Prepare JSON to query API
        wrapper function for getting site data

        @param - site_id
        @param - start_epoch
        @param - end_epoch
        @param - query_interval
        @param - query_type

        Returns raw_text
        """
        query_key = self.QUERY_ENDPOINT.format(inst_id=site_id)

        payload = {
            "type": query_type,
            "start": start_epoch,
            "end": end_epoch,
            "interval": query_interval,
        }

        logger.debug("Sending data query %s" % payload)
        data_frame = self._send_query_request(query_key, payload)
        return data_frame

    def _send_query_request(self, url, data_dict={}):
        """
        Wrapper function to add auth token for requests
        """
        response = None
        headers = {"X-Authorization": "Bearer %s" % self._auth_token}

        logger.debug("Sending data to %s" % url)
        logger.debug("Sending with headers %s" % headers)
        try:
            response = requests.get(url, headers=headers, params=data_dict)

            if response.status_code == 200:
                return response.json()
            else:
                logger.error("Something went wrong with request msg:%s" % response.text)
                return {}

            logger.debug("url: %s" % response.url)

        except Exception as e:
            logger.exception("Error with getting request")
