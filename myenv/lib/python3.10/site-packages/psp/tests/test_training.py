import tempfile
from datetime import datetime

import pandas as pd

from psp.training import PvXDataPipe
from psp.typings import Horizons


def test_pvx_datapipe(pv_data_source):

    pvx = PvXDataPipe(
        data_source=pv_data_source,
        horizons=Horizons(duration=15, num_horizons=48 * 4),
        pv_ids=["0", "1", "2"],
        start_ts=datetime(2020, 1, 1),
        end_ts=datetime(2021, 1, 1),
        step=15,
    )

    x = next(iter(pvx))
    assert x.pv_id in ["0", "1", "2"]
    assert x.ts >= datetime(2020, 1, 1)
    assert x.ts <= datetime(2021, 1, 1)


def test_pvx_datapipe_dataset(pv_data_source):

    # create a temp csv file with columns ['pv_id', 'timestamp']
    # with 2 rows of random data
    with tempfile.NamedTemporaryFile() as f:
        df = pd.DataFrame(
            {"pv_id": ["0", "1"], "timestamp": [datetime(2020, 1, 1), datetime(2020, 1, 2)]}
        )
        df.to_csv(f.name, index=False)

        pvx = PvXDataPipe(
            data_source=pv_data_source,
            horizons=Horizons(duration=15, num_horizons=48 * 4),
            pv_ids=["0", "1", "2"],
            start_ts=datetime(2020, 1, 1),
            end_ts=datetime(2021, 1, 1),
            step=15,
            dataset_file=f.name,
        )

        x, y = [x for x in iter(pvx)]
        assert x.pv_id == "0"
        assert x.ts == datetime(2020, 1, 1)

        assert y.pv_id == "1"
        assert y.ts == datetime(2020, 1, 2)
