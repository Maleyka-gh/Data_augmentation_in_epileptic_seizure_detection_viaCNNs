import pandas as pd


def get_num_of_patients_with_seizure_kind(seizure_dict):
    """Generates an overview how many patients got a specific seizure"""
    seizures = {'FAS': 0, 'FIAS': 0, 'GAS': 0, 'GMS': 0, 'TCS': 0, 'unclassified': 0}
    for patient_name, seizure_name in seizure_dict.items():
        list_of_used = []
        for seizure in seizure_name:
            s = rename(seizure['seizure_new'])
            if s not in list_of_used:
                list_of_used.append(s)
                seizures[s] = seizures[s] + 1
    return seizures


def get_num_of_seizure_and_motor_per_patient(seizures):
    """Generates an overview how many patients got a seizure (which and which motor-behave)"""
    overview = template_seizure_counter()
    for patient_name, seizure_list in seizures.items():
        list_of_used = []
        for seizure in seizure_list:
            s = rename(seizure['seizure_new'])
            m = rename(seizure['motor'])
            if (s, m) not in list_of_used:
                list_of_used.append((s, m))
                overview[s][m] = overview[s][m] + 1
    return overview


def rename(v):
    """Renames v so that it can be used as a key for the dictionary 'overview' or 'inner_dict'."""
    switcher = {'unknown': 'unclassified'}
    try:
        return switcher[v]
    except KeyError:
        return v


def template_seizure_counter():
    """Generates two nested dicts used for counting the kinds of seizures later."""
    inner_dict = {'motor': 0, 'nonmotor': 0, 'bilateraltonicclonic': 0, 'atonic': 0, 'tonic': 0, 'psychogenic': 0,
                  'unclassified': 0}
    overview = {'FAS': inner_dict.copy(),
                'FIAS': inner_dict.copy(),
                'GAS': inner_dict.copy(),
                'GMS': inner_dict.copy(),
                'TCS': inner_dict.copy(),
                'unclassified': inner_dict.copy()}
    return overview


def get_num_of_seizure_and_motor(seizures):
    """Counts the kind of seizures and structures the into 'motor' or 'non-motor' in two nested dicts."""
    # create empty nested dicts.
    overview = template_seizure_counter()
    # fill dicts
    for patient_name, seizure_list in seizures.items():
        for seizure in seizure_list:
            s = rename(seizure['seizure_new'])
            m = rename(seizure['motor'])
            overview[s][m] = overview[s][m] + 1
    return overview


def is_motor(string):
    """Checks if string is a kind of motor seizure."""
    if string == "motor" or string == "bilateraltonicclonic" or string == "atonic" or string == "tonic":
        return True
    else:
        return False




def get_motor_seizures():
    """Returns the dictionary of all motor-seizures."""
    seizures_all = get_all_seizures()
    motor = dict()
    for pat, seizures in seizures_all.items():
        motor[pat] = list()
        for s in seizures:
            if is_motor(s['motor']):
                motor[pat].append(s)
    motor = {k: v for k, v in motor.items() if v}  # delete entries in dict which are empty lists.
    return motor


def get_seizures_to_use():
    """Returns all seizures who are in the dataset. This is the method which is to be called for labeling the data."""
    all = {
        "BN_006": [dict(
            [('start', pd.Timestamp("2017-03-06 14:23:02")), ('end', pd.Timestamp("2017-03-06 14:24:33")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-03-04 14:49:57")), ('end', pd.Timestamp("2017-03-04 14:54:32")),
             ('motor', "bilateraltonicclonic"), ('seizure_old', "GTCS"), ('seizure_new', "TCS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Excel")])],
        "BN_011": [dict([('start', pd.Timestamp("2017-03-10 10:34:57")), ('end', pd.Timestamp("2017-03-10 10:36:34")),
                         ('motor', "bilateraltonicclonic"), ('seizure_old', "GTCS"), ('seizure_new', "TCS"),
                         ('onset', "temporal"), ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-03-10 18:34:26")), ('end', pd.Timestamp("2017-03-10 18:36:10")),
             ('motor', "bilateraltonicclonic"), ('seizure_old', "GTCS"), ('seizure_new', "TCS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-03-11 04:21:44")), ('end', pd.Timestamp("2017-03-11 04:23:33")),
             ('motor', "bilateraltonicclonic"), ('seizure_old', "GTCS"), ('seizure_new', "TCS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_012": [dict([('start', pd.Timestamp("2017-03-09 18:56:44")), ('end', pd.Timestamp("2017-03-09 18:57:26")),
                         ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
                         ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_016": [dict(
            [('start', pd.Timestamp("2017-03-27 10:17:52")), ('end', pd.Timestamp("2017-03-27 10:21:00")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_017": [dict(
            [('start', pd.Timestamp("2017-03-27 12:28:19")), ('end', pd.Timestamp("2017-03-27 12:28:46")),
             ('motor', "atonic"), ('seizure_old', "atonic"), ('seizure_new', "GMS"), ('onset', "generalized"),
             ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_018": [dict(
            [('start', pd.Timestamp("2017-04-22 04:40:04")), ('end', pd.Timestamp("2017-04-22 04:41:24")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-04-22 23:20:27")), ('end', pd.Timestamp("2017-04-22 23:21:53")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-04-23 02:44:28")), ('end', pd.Timestamp("2017-04-23 02:45:39")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-04-23 06:21:07")), ('end', pd.Timestamp("2017-04-23 06:22:16")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_027": [dict(
            [('start', pd.Timestamp("2017-05-11 13:27:40")), ('end', pd.Timestamp("2017-05-11 13:29:18")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-05-08 21:58:27")), ('end', pd.Timestamp("2017-05-08 21:58:56")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-05-10 05:41:00")), ('end', pd.Timestamp("2017-05-10 05:42:00")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-05-11 01:59:20")), ('end', pd.Timestamp("2017-05-11 01:59:37")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_029": [dict([('start', pd.Timestamp("2017-05-19 16:07:12")), ('end', pd.Timestamp("2017-05-19 16:08:03")),
                         ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
                         ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-05-18 18:47:22")), ('end', pd.Timestamp("2017-05-18 18:47:54")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_031": [dict([('start', pd.Timestamp("2017-05-16 20:06:56")), ('end', pd.Timestamp("2017-05-16 20:07:16")),
                         ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "frontal"),
                         ('awareness', "aware"), ('source', "Both")])],
        "BN_036": [dict([('start', pd.Timestamp("2017-05-25 22:40:30")), ('end', pd.Timestamp("2017-05-25 22:41:46")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-05-26 00:09:43")), ('end', pd.Timestamp("2017-05-26 00:10:23")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_037": [dict([('start', pd.Timestamp("2017-05-27 20:20:14")), ('end', pd.Timestamp("2017-05-27 20:21:21")),
                         ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
                         ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_041": [dict([('start', pd.Timestamp("2017-06-05 14:20:50")), ('end', pd.Timestamp("2017-06-05 14:22:23")),
                         ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
                         ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-06-05 14:56:49")), ('end', pd.Timestamp("2017-06-05 14:57:33")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-06-05 16:10:58")), ('end', pd.Timestamp("2017-06-05 16:12:02")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-06-06 01:24:40")), ('end', pd.Timestamp("2017-06-06 01:26:54")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-06-06 02:19:29")), ('end', pd.Timestamp("2017-06-06 02:20:30")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-06-06 04:46:52")), ('end', pd.Timestamp("2017-06-06 04:48:05")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-06-06 05:55:10")), ('end', pd.Timestamp("2017-06-06 05:56:24")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-06-06 06:14:51")), ('end', pd.Timestamp("2017-06-06 06:15:49")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-06-06 07:04:11")), ('end', pd.Timestamp("2017-06-06 07:04:53")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")])],
        "BN_046": [dict(
            [('start', pd.Timestamp("2017-06-20 19:01:48")), ('end', pd.Timestamp("2017-06-20 19:11:08")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_047": [dict([('start', pd.Timestamp("2017-06-21 12:34:58")), ('end', pd.Timestamp("2017-06-21 12:36:16")),
                         ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
                         ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-06-21 16:11:58")), ('end', pd.Timestamp("2017-06-21 16:12:38")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-06-21 19:16:50")), ('end', pd.Timestamp("2017-06-21 19:17:56")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-06-21 05:58:24")), ('end', pd.Timestamp("2017-06-21 06:00:03")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_051": [dict(
            [('start', pd.Timestamp("2017-07-04 05:06:46")), ('end', pd.Timestamp("2017-07-04 05:07:48")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-07-04 09:45:17")), ('end', pd.Timestamp("2017-07-04 09:46:21")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_057": [dict(
            [('start', pd.Timestamp("2017-07-10 15:59:00")), ('end', pd.Timestamp("2017-07-10 15:59:35")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "frontal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-07-10 21:41:18")), ('end', pd.Timestamp("2017-07-10 21:41:53")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "frontal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-07-10 23:02:55")), ('end', pd.Timestamp("2017-07-10 23:03:30")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "frontal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-07-11 04:04:49")), ('end', pd.Timestamp("2017-07-11 04:05:23")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "frontal"),
             ('awareness', "aware"), ('source', "Both")])],
        "BN_067": [dict(
            [('start', pd.Timestamp("2017-08-08 15:26:17")), ('end', pd.Timestamp("2017-08-08 15:27:30")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-08-08 16:58:01")), ('end', pd.Timestamp("2017-08-08 16:59:17")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-08-08 18:09:16")), ('end', pd.Timestamp("2017-08-08 18:11:04")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")]),dict(
            [('start', pd.Timestamp("2017-08-09 00:01:38")), ('end', pd.Timestamp("2017-08-09 00:02:22")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-08-09 09:32:06")), ('end', pd.Timestamp("2017-08-09 09:33:05")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")])],
        "BN_070": [dict([('start', pd.Timestamp("2017-08-27 03:34:46")), ('end', pd.Timestamp("2017-08-27 03:35:46")),
                         ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
                         ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-08-28 08:47:12")), ('end', pd.Timestamp("2017-08-28 08:47:58")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_071": [dict([('start', pd.Timestamp("2017-08-31 05:01:58")), ('end', pd.Timestamp("2017-08-31 05:02:27")),
                         ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
                         ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-09-04 01:45:11")), ('end', pd.Timestamp("2017-09-04 01:49:02")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_072": [dict([('start', pd.Timestamp("2017-09-13 13:27:04")), ('end', pd.Timestamp("2017-09-13 13:27:39")),
                         ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
                         ('awareness', "aware"), ('source', "Both")])],
        "BN_082": [dict([('start', pd.Timestamp("2017-10-18 14:01:28")), ('end', pd.Timestamp("2017-10-18 14:02:43")),
                         ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
                         ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-10-19 10:01:07")), ('end', pd.Timestamp("2017-10-19 10:02:49")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-10-20 12:57:48")), ('end', pd.Timestamp("2017-10-20 13:00:14")),
             ('motor', "bilateraltonicclonic"), ('seizure_old', "GTCS"), ('seizure_new', "TCS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_083": [dict([('start', pd.Timestamp("2017-10-22 07:32:27")), ('end', pd.Timestamp("2017-10-22 07:35:34")),
                         ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "hemispheric"),
                         ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-10-24 00:29:50")), ('end', pd.Timestamp("2017-10-24 00:36:35")),
             ('motor', "bilateraltonicclonic"), ('seizure_old', "GTCS"), ('seizure_new', "TCS"),
             ('onset', "hemispheric"), ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_084": [dict([('start', pd.Timestamp("2017-10-18 12:49:03")), ('end', pd.Timestamp("2017-10-18 12:49:19")),
                         ('motor', "tonic"), ('seizure_old', "tonic"), ('seizure_new', "GMS"), ('onset', "generalized"),
                         ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_086": [dict([('start', pd.Timestamp("2017-10-27 00:53:45")), ('end', pd.Timestamp("2017-10-27 00:54:57")),
                         ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
                         ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-10-27 07:34:08")), ('end', pd.Timestamp("2017-10-27 07:35:17")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-10-26 08:15:02")), ('end', pd.Timestamp("2017-10-26 08:17:31")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_100": [dict([('start', pd.Timestamp("2017-12-14 10:03:54")), ('end', pd.Timestamp("2017-12-14 10:06:14")),
                         ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
                         ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_103": [dict([('start', pd.Timestamp("2018-01-02 15:39:47")), ('end', pd.Timestamp("2018-01-02 15:41:20")),
                         ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
                         ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-01-02 21:15:54")), ('end', pd.Timestamp("2018-01-02 21:16:39")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_106": [dict([('start', pd.Timestamp("2018-01-24 09:43:07")), ('end', pd.Timestamp("2018-01-24 09:44:08")),
                         ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "hemispheric"),
                         ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_107": [dict(
            [('start', pd.Timestamp("2018-01-25 14:57:02")), ('end', pd.Timestamp("2018-01-25 14:58:26")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_113": [dict([('start', pd.Timestamp("2018-02-07 11:04:33")), ('end', pd.Timestamp("2018-02-07 11:06:01")),
                         ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
                         ('awareness', "aware"), ('source', "Both")])],
        "BN_123": [dict([('start', pd.Timestamp("2018-03-25 03:30:18")), ('end', pd.Timestamp("2018-03-25 03:30:55")),
                         ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
                         ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-03-25 16:49:42")), ('end', pd.Timestamp("2018-03-25 16:51:15")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-03-25 18:43:13")), ('end', pd.Timestamp("2018-03-25 18:44:06")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-03-25 20:14:05")), ('end', pd.Timestamp("2018-03-25 20:14:57")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-03-25 21:14:48")), ('end', pd.Timestamp("2018-03-25 21:15:48")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-03-25 22:02:44")), ('end', pd.Timestamp("2018-03-25 22:03:53")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-03-25 23:25:28")), ('end', pd.Timestamp("2018-03-25 23:26:52")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_124": [dict(
            [('start', pd.Timestamp("2018-04-21 17:33:46")), ('end', pd.Timestamp("2018-04-21 17:35:45")),
             ('motor', "bilateraltonicclonic"), ('seizure_old', "GTCS"), ('seizure_new', "TCS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_138": [dict([('start', pd.Timestamp("2018-06-05 11:48:38")), ('end', pd.Timestamp("2018-06-05 11:49:08")),
                         ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
                         ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_139": [dict(
            [('start', pd.Timestamp("2018-06-11 23:11:24")), ('end', pd.Timestamp("2018-06-11 23:13:52")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Excel")])],
        "BN_141": [dict([('start', pd.Timestamp("2018-06-12 11:37:09")), ('end', pd.Timestamp("2018-06-12 11:38:37")),
                         ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
                         ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-06-12 06:30:18")), ('end', pd.Timestamp("2018-06-12 06:31:44")),
             ('motor', "bilateraltonicclonic"), ('seizure_old', "GTCS"), ('seizure_new', "TCS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_146": [dict([('start', pd.Timestamp("2018-07-02 04:40:14")), ('end', pd.Timestamp("2018-07-02 04:42:14")),
                         ('motor', "bilateraltonicclonic"), ('seizure_old', "GTCS"), ('seizure_new', "TCS"),
                         ('onset', "temporal"), ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_149": [dict([('start', pd.Timestamp("2018-07-12 16:18:44")), ('end', pd.Timestamp("2018-07-12 16:19:30")),
                         ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "frontal"),
                         ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-07-12 17:23:11")), ('end', pd.Timestamp("2018-07-12 17:23:45")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "frontal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-07-12 18:20:23")), ('end', pd.Timestamp("2018-07-12 18:20:57")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "frontal"),
             ('awareness', "aware"), ('source', "Both")])],
        "BN_159": [dict(
            [('start', pd.Timestamp("2018-08-17 06:05:10")), ('end', pd.Timestamp("2018-08-17 06:06:07")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_160": [dict(
            [('start', pd.Timestamp("2018-08-24 09:51:43")), ('end', pd.Timestamp("2018-08-24 09:52:03")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-08-24 09:53:51")), ('end', pd.Timestamp("2018-08-24 09:54:18")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-08-24 09:55:47")), ('end', pd.Timestamp("2018-08-24 09:56:08")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-08-24 10:00:26")), ('end', pd.Timestamp("2018-08-24 10:00:53")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")])],
        "BN_166": [dict([('start', pd.Timestamp("2018-10-15 16:08:20")), ('end', pd.Timestamp("2018-10-15 16:09:52")),
                         ('motor', "bilateraltonicclonic"), ('seizure_old', "GTCS"), ('seizure_new', "TCS"),
                         ('onset', "temporal"), ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_167": [dict([('start', pd.Timestamp("2018-10-19 05:56:16")), ('end', pd.Timestamp("2018-10-19 05:56:58")),
                         ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "unknown"),
                         ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-10-19 05:08:24")), ('end', pd.Timestamp("2018-10-19 05:09:39")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "unknown"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-10-19 06:17:04")), ('end', pd.Timestamp("2018-10-19 06:17:13")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "unknown"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-10-19 06:29:09")), ('end', pd.Timestamp("2018-10-19 06:29:56")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "unknown"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-10-19 06:31:32")), ('end', pd.Timestamp("2018-10-19 06:32:03")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "unknown"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-10-19 06:36:43")), ('end', pd.Timestamp("2018-10-19 06:37:15")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "unknown"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-10-19 06:38:53")), ('end', pd.Timestamp("2018-10-19 06:39:34")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "unknown"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-10-19 06:46:56")), ('end', pd.Timestamp("2018-10-19 06:47:19")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "unknown"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-10-19 06:49:23")), ('end', pd.Timestamp("2018-10-19 06:49:59")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "unknown"),
             ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_169": [dict([('start', pd.Timestamp("2018-11-09 12:46:59")), ('end', pd.Timestamp("2018-11-09 12:47:34")),
                         ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
                         ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-11-09 13:35:25")), ('end', pd.Timestamp("2018-11-09 13:36:07")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")])],
        "BN_170": [dict([('start', pd.Timestamp("2018-11-13 17:07:15")), ('end', pd.Timestamp("2018-11-13 17:08:38")),
                         ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
                         ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-11-13 21:55:53")), ('end', pd.Timestamp("2018-11-13 21:57:59")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-11-14 07:32:05")), ('end', pd.Timestamp("2018-11-14 07:35:53")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_179": [dict([('start', pd.Timestamp("2019-02-07 08:51:33")), ('end', pd.Timestamp("2019-02-07 08:53:24")),
                         ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
                         ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2019-02-04 19:40:11")), ('end', pd.Timestamp("2019-02-04 19:52:37")),
             ('motor', "bilateraltonicclonic"), ('seizure_old', "GTCS"), ('seizure_new', "TCS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_180": [dict(
            [('start', pd.Timestamp("2019-02-20 07:31:02")), ('end', pd.Timestamp("2019-02-20 07:32:45")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_186": [dict(
            [('start', pd.Timestamp("2019-04-14 22:03:27")), ('end', pd.Timestamp("2019-04-14 22:05:58")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2019-04-15 04:55:05")), ('end', pd.Timestamp("2019-04-15 04:57:21")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2019-04-15 06:46:48")), ('end', pd.Timestamp("2019-04-15 06:47:56")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_199": [dict([('start', pd.Timestamp("2019-07-07 15:36:06")), ('end', pd.Timestamp("2019-07-07 15:37:41")),
                         ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
                         ('awareness', "aware"), ('source', "Both")])]
    }
    return all


def get_all_seizures():
    """dictionary of all seizures in meta-files and excel-overview per patient."""
    all = {
        "BN_006": [dict([('start', pd.Timestamp("2017-03-01 17:48:50")), ('end', pd.Timestamp("2017-03-01 17:50:05")),
                         ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
                         ('awareness', "impairedawareness"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2017-03-06 14:23:02")), ('end', pd.Timestamp("2017-03-06 14:24:33")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-03-04 14:49:57")), ('end', pd.Timestamp("2017-03-04 14:54:32")),
             ('motor', "bilateraltonicclonic"), ('seizure_old', "GTCS"), ('seizure_new', "TCS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Excel")])],
        "BN_009": [dict([('start', pd.Timestamp("2017-03-08 14:18:28")), ('end', pd.Timestamp("2017-03-08 14:19:13")),
                         ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
                         ('awareness', "impairedawareness"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2017-03-08 15:59:40")), ('end', pd.Timestamp("2017-03-08 16:02:00")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Excel")])],
        "BN_010": [],
        "BN_011": [dict([('start', pd.Timestamp("2017-03-10 10:34:57")), ('end', pd.Timestamp("2017-03-10 10:36:34")),
                         ('motor', "bilateraltonicclonic"), ('seizure_old', "GTCS"), ('seizure_new', "TCS"),
                         ('onset', "temporal"), ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-03-10 18:34:26")), ('end', pd.Timestamp("2017-03-10 18:36:10")),
             ('motor', "bilateraltonicclonic"), ('seizure_old', "GTCS"), ('seizure_new', "TCS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-03-11 04:21:44")), ('end', pd.Timestamp("2017-03-11 04:23:33")),
             ('motor', "bilateraltonicclonic"), ('seizure_old', "GTCS"), ('seizure_new', "TCS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_012": [dict([('start', pd.Timestamp("2017-03-09 18:56:44")), ('end', pd.Timestamp("2017-03-09 18:57:26")),
                         ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
                         ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-03-13 23:23:40")), ('end', pd.Timestamp("2017-03-13 23:25:15")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Excel")])],
        "BN_014": [dict([('start', pd.Timestamp("2017-03-16 22:07:25")), ('end', pd.Timestamp("2017-03-16 22:09:43")),
                         ('motor', "nonmotor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
                         ('awareness', "aware"), ('source', "Both")])],
        "BN_016": [dict([('start', pd.Timestamp("2017-03-24 07:55:54")), ('end', pd.Timestamp("2017-03-24 07:56:42")),
                         ('motor', "nonmotor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
                         ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-03-25 15:45:18")), ('end', pd.Timestamp("2017-03-25 15:45:49")),
             ('motor', "nonmotor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2017-03-27 12:14:32")), ('end', pd.Timestamp("2017-03-27 12:15:05")),
             ('motor', "nonmotor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-03-23 08:28:44")), ('end', pd.Timestamp("2017-03-23 08:30:44")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2017-03-25 08:44:45")), ('end', pd.Timestamp("2017-03-25 08:46:53")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2017-03-27 10:17:52")), ('end', pd.Timestamp("2017-03-27 10:21:00")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_017": [dict([('start', pd.Timestamp("2017-03-27 14:35:50")), ('end', pd.Timestamp("2017-03-27 14:36:11")),
                         ('motor', "nonmotor"), ('seizure_old', "atypicalAbsence"), ('seizure_new', "GAS"),
                         ('onset', "generalized"), ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-03-27 12:28:19")), ('end', pd.Timestamp("2017-03-27 12:28:46")),
             ('motor', "atonic"), ('seizure_old', "atonic"), ('seizure_new', "GMS"), ('onset', "generalized"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-03-27 20:02:55")), ('end', pd.Timestamp("2017-03-27 20:03:13")),
             ('motor', "atonic"), ('seizure_old', "atonic"), ('seizure_new', "GMS"), ('onset', "generalized"),
             ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2017-03-27 11:22:04")), ('end', pd.Timestamp("2017-03-27 11:22:17")),
             ('motor', "unclassified"), ('seizure_old', "subclinicalEEGonly"), ('seizure_new', "unclassified"),
             ('onset', "generalized"), ('awareness', "unclassified"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-03-27 12:30:04")), ('end', pd.Timestamp("2017-03-27 12:30:15")),
             ('motor', "unclassified"), ('seizure_old', "subclinicalEEGonly"), ('seizure_new', "unclassified"),
             ('onset', "generalized"), ('awareness', "unclassified"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-03-27 13:12:56")), ('end', pd.Timestamp("2017-03-27 13:13:06")),
             ('motor', "unclassified"), ('seizure_old', "subclinicalEEGonly"), ('seizure_new', "unclassified"),
             ('onset', "generalized"), ('awareness', "unclassified"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-03-27 13:25:26")), ('end', pd.Timestamp("2017-03-27 13:25:38")),
             ('motor', "unclassified"), ('seizure_old', "subclinicalEEGonly"), ('seizure_new', "unclassified"),
             ('onset', "generalized"), ('awareness', "unclassified"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-03-27 13:27:43")), ('end', pd.Timestamp("2017-03-27 13:27:53")),
             ('motor', "unclassified"), ('seizure_old', "subclinicalEEGonly"), ('seizure_new', "unclassified"),
             ('onset', "generalized"), ('awareness', "unclassified"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-03-27 13:35:57")), ('end', pd.Timestamp("2017-03-27 13:36:09")),
             ('motor', "unclassified"), ('seizure_old', "subclinicalEEGonly"), ('seizure_new', "unclassified"),
             ('onset', "generalized"), ('awareness', "unclassified"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-03-27 16:08:48")), ('end', pd.Timestamp("2017-03-27 16:09:00")),
             ('motor', "nonmotor"), ('seizure_old', "subclinicalEEGonly"), ('seizure_new', "unclassified"),
             ('onset', "generalized"), ('awareness', "unclassified"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2017-03-27 20:06:18")), ('end', pd.Timestamp("2017-03-27 20:06:31")),
             ('motor', "nonmotor"), ('seizure_old', "subclinicalEEGonly"), ('seizure_new', "unclassified"),
             ('onset', "generalized"), ('awareness', "unclassified"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2017-03-28 01:00:41")), ('end', pd.Timestamp("2017-03-28 01:00:53")),
             ('motor', "nonmotor"), ('seizure_old', "subclinicalEEGonly"), ('seizure_new', "unclassified"),
             ('onset', "generalized"), ('awareness', "unclassified"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2017-03-28 04:01:37")), ('end', pd.Timestamp("2017-03-28 04:01:47")),
             ('motor', "nonmotor"), ('seizure_old', "subclinicalEEGonly"), ('seizure_new', "unclassified"),
             ('onset', "generalized"), ('awareness', "unclassified"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2017-03-28 06:52:15")), ('end', pd.Timestamp("2017-03-28 06:52:35")),
             ('motor', "nonmotor"), ('seizure_old', "subclinicalEEGonly"), ('seizure_new', "unclassified"),
             ('onset', "generalized"), ('awareness', "unclassified"), ('source', "Excel")])],
        "BN_018": [dict([('start', pd.Timestamp("2017-04-22 19:17:23")), ('end', pd.Timestamp("2017-04-22 19:17:49")),
                         ('motor', "nonmotor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
                         ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-04-22 23:04:59")), ('end', pd.Timestamp("2017-04-22 23:05:44")),
             ('motor', "nonmotor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-04-23 10:15:50")), ('end', pd.Timestamp("2017-04-23 10:16:44")),
             ('motor', "nonmotor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2017-04-22 04:40:04")), ('end', pd.Timestamp("2017-04-22 04:41:24")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-04-22 23:20:27")), ('end', pd.Timestamp("2017-04-22 23:21:53")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-04-23 02:44:28")), ('end', pd.Timestamp("2017-04-23 02:45:39")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-04-23 06:21:07")), ('end', pd.Timestamp("2017-04-23 06:22:16")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-04-23 13:33:09")), ('end', pd.Timestamp("2017-04-23 13:34:08")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Excel")])],
        "BN_027": [dict([('start', pd.Timestamp("2017-05-09 13:35:23")), ('end', pd.Timestamp("2017-05-09 13:35:33")),
                         ('motor', "nonmotor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
                         ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-05-09 10:24:57")), ('end', pd.Timestamp("2017-05-09 10:26:12")),
             ('motor', "nonmotor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-05-10 10:18:00")), ('end', pd.Timestamp("2017-05-10 10:18:29")),
             ('motor', "nonmotor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-05-11 13:27:40")), ('end', pd.Timestamp("2017-05-11 13:29:18")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-05-08 21:58:27")), ('end', pd.Timestamp("2017-05-08 21:58:56")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-05-10 05:41:00")), ('end', pd.Timestamp("2017-05-10 05:42:00")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-05-11 01:59:20")), ('end', pd.Timestamp("2017-05-11 01:59:37")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_029": [dict([('start', pd.Timestamp("2017-05-19 16:07:12")), ('end', pd.Timestamp("2017-05-19 16:08:03")),
                         ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
                         ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-05-18 18:47:22")), ('end', pd.Timestamp("2017-05-18 18:47:54")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_030": [],
        "BN_031": [dict([('start', pd.Timestamp("2017-05-16 20:06:56")), ('end', pd.Timestamp("2017-05-16 20:07:16")),
                         ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "frontal"),
                         ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-05-16 22:33:18")), ('end', pd.Timestamp("2017-05-16 22:33:40")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "frontal"),
             ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2017-05-16 23:59:55")), ('end', pd.Timestamp("2017-05-17 00:00:25")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "frontal"),
             ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2017-05-17 01:31:16")), ('end', pd.Timestamp("2017-05-17 01:31:45")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "frontal"),
             ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2017-05-17 03:34:06")), ('end', pd.Timestamp("2017-05-17 03:34:25")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "frontal"),
             ('awareness', "aware"), ('source', "Excel")])],
        "BN_034": [dict([('start', pd.Timestamp("2017-05-28 00:36:24")), ('end', pd.Timestamp("2017-05-28 00:39:48")),
                         ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
                         ('awareness', "impairedawareness"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2017-05-28 03:55:23")), ('end', pd.Timestamp("2017-05-28 03:59:24")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2017-05-28 04:53:49")), ('end', pd.Timestamp("2017-05-28 04:55:09")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2017-05-28 05:27:30")), ('end', pd.Timestamp("2017-05-28 05:28:38")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2017-05-28 05:58:56")), ('end', pd.Timestamp("2017-05-28 06:02:42")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2017-05-29 00:09:06")), ('end', pd.Timestamp("2017-05-29 00:11:01")),
             ('motor', "nonmotor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2017-05-29 02:18:51")), ('end', pd.Timestamp("2017-05-29 02:20:22")),
             ('motor', "nonmotor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Excel")])],
        "BN_036": [dict([('start', pd.Timestamp("2017-05-24 21:21:47")), ('end', pd.Timestamp("2017-05-24 21:25:46")),
                         ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
                         ('awareness', "impairedawareness"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2017-05-25 22:40:30")), ('end', pd.Timestamp("2017-05-25 22:41:46")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-05-26 00:09:43")), ('end', pd.Timestamp("2017-05-26 00:10:23")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_037": [dict([('start', pd.Timestamp("2017-05-27 20:20:14")), ('end', pd.Timestamp("2017-05-27 20:21:21")),
                         ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
                         ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-05-26 23:18:54")), ('end', pd.Timestamp("2017-05-26 23:19:55")),
             ('motor', "unclassified"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "unclassified"), ('source', "Both")])],
        "BN_040": [dict([('start', pd.Timestamp("2017-06-02 03:05:02")), ('end', pd.Timestamp("2017-06-02 03:06:09")),
                         ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
                         ('awareness', "impairedawareness"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2017-06-02 13:50:21")), ('end', pd.Timestamp("2017-06-02 13:51:53")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2017-06-02 16:20:50")), ('end', pd.Timestamp("2017-06-02 16:22:51")),
             ('motor', "bilateraltonicclonic"), ('seizure_old', "GTCS"), ('seizure_new', "TCS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Excel")])],
        "BN_041": [dict([('start', pd.Timestamp("2017-06-05 14:20:50")), ('end', pd.Timestamp("2017-06-05 14:22:23")),
                         ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
                         ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-06-05 14:56:49")), ('end', pd.Timestamp("2017-06-05 14:57:33")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-06-05 16:10:58")), ('end', pd.Timestamp("2017-06-05 16:12:02")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-06-06 01:24:40")), ('end', pd.Timestamp("2017-06-06 01:26:54")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-06-06 02:19:29")), ('end', pd.Timestamp("2017-06-06 02:20:30")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-06-06 03:38:50")), ('end', pd.Timestamp("2017-06-06 03:39:47")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2017-06-06 04:46:52")), ('end', pd.Timestamp("2017-06-06 04:48:05")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-06-06 05:55:10")), ('end', pd.Timestamp("2017-06-06 05:56:24")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-06-06 06:14:51")), ('end', pd.Timestamp("2017-06-06 06:15:49")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-06-06 07:04:11")), ('end', pd.Timestamp("2017-06-06 07:04:53")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")])],
        "BN_042": [dict([('start', pd.Timestamp("2017-05-31 16:33:57")), ('end', pd.Timestamp("2017-05-31 16:43:06")),
                         ('motor', "nonmotor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "unknown"),
                         ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-05-31 18:16:59")), ('end', pd.Timestamp("2017-05-31 18:24:49")),
             ('motor', "nonmotor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "unknown"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-06-01 16:14:11")), ('end', pd.Timestamp("2017-06-01 16:22:11")),
             ('motor', "nonmotor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "unknown"),
             ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2017-06-01 20:01:15")), ('end', pd.Timestamp("2017-06-01 20:07:22")),
             ('motor', "nonmotor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "unknown"),
             ('awareness', "aware"), ('source', "Excel")])],
        "BN_044": [],
        "BN_045": [dict([('start', pd.Timestamp("2017-06-13 16:42:37")), ('end', pd.Timestamp("2017-06-13 16:44:18")),
                         ('motor', "nonmotor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
                         ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-06-14 00:22:21")), ('end', pd.Timestamp("2017-06-14 00:24:10")),
             ('motor', "nonmotor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-06-13 19:23:44")), ('end', pd.Timestamp("2017-06-13 19:27:04")),
             ('motor', "nonmotor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2017-06-14 04:37:19")), ('end', pd.Timestamp("2017-06-14 04:38:31")),
             ('motor', "nonmotor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")])],
        "BN_046": [dict([('start', pd.Timestamp("2017-06-19 17:57:54")), ('end', pd.Timestamp("2017-06-19 17:59:05")),
                         ('motor', "nonmotor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
                         ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-06-20 19:01:48")), ('end', pd.Timestamp("2017-06-20 19:11:08")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_047": [dict([('start', pd.Timestamp("2017-06-21 12:34:58")), ('end', pd.Timestamp("2017-06-21 12:36:16")),
                         ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
                         ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-06-21 16:11:58")), ('end', pd.Timestamp("2017-06-21 16:12:38")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-06-21 19:16:50")), ('end', pd.Timestamp("2017-06-21 19:17:56")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-06-21 20:44:15")), ('end', pd.Timestamp("2017-06-21 20:44:21")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2017-06-21 05:58:24")), ('end', pd.Timestamp("2017-06-21 06:00:03")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_048": [],
        "BN_051": [dict([('start', pd.Timestamp("2017-07-04 01:28:34")), ('end', pd.Timestamp("2017-07-04 01:29:05")),
                         ('motor', "nonmotor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
                         ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2017-07-04 03:04:25")), ('end', pd.Timestamp("2017-07-04 03:05:38")),
             ('motor', "nonmotor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2017-07-04 05:06:46")), ('end', pd.Timestamp("2017-07-04 05:07:48")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-07-04 09:45:17")), ('end', pd.Timestamp("2017-07-04 09:46:21")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_052": [dict([('start', pd.Timestamp("2017-07-04 05:08:20")), ('end', pd.Timestamp("2017-07-04 05:09:20")),
                         ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
                         ('awareness', "impairedawareness"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2017-07-04 06:58:29")), ('end', pd.Timestamp("2017-07-04 06:59:35")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Excel")])],
        "BN_054": [],
        "BN_057": [dict([('start', pd.Timestamp("2017-07-10 14:15:30")), ('end', pd.Timestamp("2017-07-10 14:15:43")),
                         ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "frontal"),
                         ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2017-07-10 14:39:37")), ('end', pd.Timestamp("2017-07-10 14:40:15")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "frontal"),
             ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2017-07-10 15:03:26")), ('end', pd.Timestamp("2017-07-10 15:03:48")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "frontal"),
             ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2017-07-10 15:13:21")), ('end', pd.Timestamp("2017-07-10 15:13:54")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "frontal"),
             ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2017-07-10 15:59:00")), ('end', pd.Timestamp("2017-07-10 15:59:35")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "frontal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-07-10 21:41:18")), ('end', pd.Timestamp("2017-07-10 21:41:53")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "frontal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-07-10 23:02:55")), ('end', pd.Timestamp("2017-07-10 23:03:30")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "frontal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-07-11 04:04:49")), ('end', pd.Timestamp("2017-07-11 04:05:23")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "frontal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-07-11 07:58:15")), ('end', pd.Timestamp("2017-07-11 07:58:48")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "frontal"),
             ('awareness', "aware"), ('source', "Excel")])],
        "BN_067": [dict([('start', pd.Timestamp("2017-08-08 14:05:40")), ('end', pd.Timestamp("2017-08-08 14:07:07")),
                         ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
                         ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2017-08-08 15:26:17")), ('end', pd.Timestamp("2017-08-08 15:27:30")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-08-08 16:58:01")), ('end', pd.Timestamp("2017-08-08 16:59:17")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-08-08 18:09:16")), ('end', pd.Timestamp("2017-08-08 18:11:04")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-08-08 21:12:02")), ('end', pd.Timestamp("2017-08-08 21:13:30")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2017-08-09 00:01:38")), ('end', pd.Timestamp("2017-08-09 00:02:22")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-08-09 09:32:06")), ('end', pd.Timestamp("2017-08-09 09:33:05")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-08-09 10:26:57")), ('end', pd.Timestamp("2017-08-09 10:28:08")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Excel")])],
        "BN_070": [dict([('start', pd.Timestamp("2017-08-27 03:34:46")), ('end', pd.Timestamp("2017-08-27 03:35:46")),
                         ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
                         ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-08-28 08:47:12")), ('end', pd.Timestamp("2017-08-28 08:47:58")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_071": [dict([('start', pd.Timestamp("2017-08-31 05:01:58")), ('end', pd.Timestamp("2017-08-31 05:02:27")),
                         ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
                         ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-09-04 01:45:11")), ('end', pd.Timestamp("2017-09-04 01:49:02")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_072": [dict([('start', pd.Timestamp("2017-09-13 13:27:04")), ('end', pd.Timestamp("2017-09-13 13:27:39")),
                         ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
                         ('awareness', "aware"), ('source', "Both")])],
        "BN_076": [],
        "BN_080": [],
        "BN_082": [dict([('start', pd.Timestamp("2017-10-18 14:01:28")), ('end', pd.Timestamp("2017-10-18 14:02:43")),
                         ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
                         ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-10-19 10:01:07")), ('end', pd.Timestamp("2017-10-19 10:02:49")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-10-20 12:57:48")), ('end', pd.Timestamp("2017-10-20 13:00:14")),
             ('motor', "bilateraltonicclonic"), ('seizure_old', "GTCS"), ('seizure_new', "TCS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_083": [dict([('start', pd.Timestamp("2017-10-22 07:32:27")), ('end', pd.Timestamp("2017-10-22 07:35:34")),
                         ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "hemispheric"),
                         ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-10-24 00:29:50")), ('end', pd.Timestamp("2017-10-24 00:36:35")),
             ('motor', "bilateraltonicclonic"), ('seizure_old', "GTCS"), ('seizure_new', "TCS"),
             ('onset', "hemispheric"), ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_084": [dict([('start', pd.Timestamp("2017-10-18 12:49:03")), ('end', pd.Timestamp("2017-10-18 12:49:19")),
                         ('motor', "tonic"), ('seizure_old', "tonic"), ('seizure_new', "GMS"), ('onset', "generalized"),
                         ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_086": [dict([('start', pd.Timestamp("2017-10-27 00:53:45")), ('end', pd.Timestamp("2017-10-27 00:54:57")),
                         ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
                         ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-10-27 07:34:08")), ('end', pd.Timestamp("2017-10-27 07:35:17")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-10-26 08:15:02")), ('end', pd.Timestamp("2017-10-26 08:17:31")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_090": [],
        "BN_091": [dict([('start', pd.Timestamp("2017-11-12 00:27:33")), ('end', pd.Timestamp("2017-11-12 00:28:32")),
                         ('motor', "nonmotor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
                         ('awareness', "impairedawareness"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2017-11-12 06:32:26")), ('end', pd.Timestamp("2017-11-12 06:34:37")),
             ('motor', "nonmotor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-11-12 09:03:23")), ('end', pd.Timestamp("2017-11-12 09:04:17")),
             ('motor', "nonmotor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_097": [dict([('start', pd.Timestamp("2017-12-08 10:57:22")), ('end', pd.Timestamp("2017-12-08 11:01:54")),
                         ('motor', "bilateraltonicclonic"), ('seizure_old', "GTCS"), ('seizure_new', "TCS"),
                         ('onset', "frontal"), ('awareness', "impairedawareness"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2017-12-08 13:55:00")), ('end', pd.Timestamp("2017-12-08 13:56:26")),
             ('motor', "bilateraltonicclonic"), ('seizure_old', "GTCS"), ('seizure_new', "TCS"), ('onset', "frontal"),
             ('awareness', "impairedawareness"), ('source', "Excel")])],
        "BN_100": [dict([('start', pd.Timestamp("2017-12-14 10:03:54")), ('end', pd.Timestamp("2017-12-14 10:06:14")),
                         ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
                         ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2017-12-17 19:16:18")), ('end', pd.Timestamp("2017-12-17 19:18:09")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2017-12-19 00:58:08")), ('end', pd.Timestamp("2017-12-19 01:00:45")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Excel")])],
        "BN_103": [dict([('start', pd.Timestamp("2018-01-02 15:39:47")), ('end', pd.Timestamp("2018-01-02 15:41:20")),
                         ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
                         ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-01-02 21:15:54")), ('end', pd.Timestamp("2018-01-02 21:16:39")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-01-03 05:29:25")), ('end', pd.Timestamp("2018-01-03 05:30:32")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Excel")])],
        "BN_104": [dict([('start', pd.Timestamp("2018-01-14 21:43:57")), ('end', pd.Timestamp("2018-01-14 21:44:42")),
                         ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
                         ('awareness', "impairedawareness"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2018-01-14 16:59:29")), ('end', pd.Timestamp("2018-01-14 17:00:55")),
             ('motor', "bilateraltonicclonic"), ('seizure_old', "GTCS"), ('seizure_new', "TCS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Excel")])],
        "BN_105": [],
        "BN_106": [dict([('start', pd.Timestamp("2018-01-24 09:43:07")), ('end', pd.Timestamp("2018-01-24 09:44:08")),
                         ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "hemispheric"),
                         ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-01-24 00:45:00")), ('end', pd.Timestamp("2018-01-24 00:46:20")),
             ('motor', "unclassified"), ('seizure_old', "subclinicalEEGonly"), ('seizure_new', "unclassified"),
             ('onset', "frontal"), ('awareness', "unclassified"), ('source', "Both")])],
        "BN_107": [dict([('start', pd.Timestamp("2018-01-25 08:46:29")), ('end', pd.Timestamp("2018-01-25 08:47:16")),
                         ('motor', "nonmotor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
                         ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2018-01-25 14:57:02")), ('end', pd.Timestamp("2018-01-25 14:58:26")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_108": [dict([('start', pd.Timestamp("2018-01-27 23:39:22")), ('end', pd.Timestamp("2018-01-27 23:40:32")),
                         ('motor', "nonmotor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
                         ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_110": [],
        "BN_112": [],
        "BN_113": [dict([('start', pd.Timestamp("2018-02-07 11:04:33")), ('end', pd.Timestamp("2018-02-07 11:06:01")),
                         ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
                         ('awareness', "aware"), ('source', "Both")])],
        "BN_116": [],
        "BN_118": [dict([('start', pd.Timestamp("2017-03-11 00:24:14")), ('end', pd.Timestamp("2017-03-11 00:25:00")),
                         ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "frontal"),
                         ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2017-03-11 03:58:08")), ('end', pd.Timestamp("2017-03-11 03:59:01")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "frontal"),
             ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2017-03-11 06:15:37")), ('end', pd.Timestamp("2017-03-11 06:16:19")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "frontal"),
             ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2017-03-12 02:20:37")), ('end', pd.Timestamp("2017-03-12 02:21:47")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "frontal"),
             ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2017-03-12 04:29:52")), ('end', pd.Timestamp("2017-03-12 04:30:43")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "frontal"),
             ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2017-03-12 06:33:56")), ('end', pd.Timestamp("2017-03-12 06:34:40")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "frontal"),
             ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2017-03-10 06:12:12")), ('end', pd.Timestamp("2017-03-10 06:13:02")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "frontal"),
             ('awareness', "impairedawareness"), ('source', "Excel")])],
        "BN_123": [dict([('start', pd.Timestamp("2018-03-25 03:30:18")), ('end', pd.Timestamp("2018-03-25 03:30:55")),
                         ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
                         ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-03-25 16:49:42")), ('end', pd.Timestamp("2018-03-25 16:51:15")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-03-25 18:43:13")), ('end', pd.Timestamp("2018-03-25 18:44:06")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-03-25 20:14:05")), ('end', pd.Timestamp("2018-03-25 20:14:57")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-03-25 21:14:48")), ('end', pd.Timestamp("2018-03-25 21:15:48")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-03-25 22:02:44")), ('end', pd.Timestamp("2018-03-25 22:03:53")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-03-25 23:25:28")), ('end', pd.Timestamp("2018-03-25 23:26:52")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_124": [dict([('start', pd.Timestamp("2018-04-19 09:57:33")), ('end', pd.Timestamp("2018-04-19 09:58:09")),
                         ('motor', "nonmotor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
                         ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-04-21 11:16:57")), ('end', pd.Timestamp("2018-04-21 11:18:00")),
             ('motor', "nonmotor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2018-04-21 16:45:17")), ('end', pd.Timestamp("2018-04-21 16:46:50")),
             ('motor', "nonmotor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-04-16 15:21:15")), ('end', pd.Timestamp("2018-04-16 15:23:10")),
             ('motor', "nonmotor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-04-21 17:33:46")), ('end', pd.Timestamp("2018-04-21 17:35:45")),
             ('motor', "bilateraltonicclonic"), ('seizure_old', "GTCS"), ('seizure_new', "TCS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-04-21 13:08:15")), ('end', pd.Timestamp("2018-04-21 13:08:40")),
             ('motor', "unclassified"), ('seizure_old', "subclinicalEEGonly"), ('seizure_new', "unclassified"),
             ('onset', "temporal"), ('awareness', "unclassified"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-04-21 17:03:32")), ('end', pd.Timestamp("2018-04-21 17:03:58")),
             ('motor', "unclassified"), ('seizure_old', "subclinicalEEGonly"), ('seizure_new', "unclassified"),
             ('onset', "temporal"), ('awareness', "unclassified"), ('source', "Both")])],
        "BN_129": [dict([('start', pd.Timestamp("2018-05-10 13:32:49")), ('end', pd.Timestamp("2018-05-10 13:33:56")),
                         ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
                         ('awareness', "impairedawareness"), ('source', "Excel")])],
        "BN_130": [],
        "BN_137": [],
        "BN_138": [dict([('start', pd.Timestamp("2018-06-05 11:48:38")), ('end', pd.Timestamp("2018-06-05 11:49:08")),
                         ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
                         ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-06-05 17:56:20")), ('end', pd.Timestamp("2018-06-05 17:56:54")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Excel")])],
        "BN_139": [dict([('start', pd.Timestamp("2018-06-11 09:28:37")), ('end', pd.Timestamp("2018-06-11 09:33:25")),
                         ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
                         ('awareness', "impairedawareness"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2018-06-11 23:11:24")), ('end', pd.Timestamp("2018-06-11 23:13:52")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2018-06-12 02:46:36")), ('end', pd.Timestamp("2018-06-12 02:49:26")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2018-06-12 09:57:13")), ('end', pd.Timestamp("2018-06-12 10:03:40")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Excel")])],
        "BN_141": [dict([('start', pd.Timestamp("2018-06-12 11:37:09")), ('end', pd.Timestamp("2018-06-12 11:38:37")),
                         ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
                         ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-06-12 06:30:18")), ('end', pd.Timestamp("2018-06-12 06:31:44")),
             ('motor', "bilateraltonicclonic"), ('seizure_old', "GTCS"), ('seizure_new', "TCS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_144": [dict([('start', pd.Timestamp("2018-06-22 07:45:38")), ('end', pd.Timestamp("2018-06-22 07:45:41")),
                         ('motor', "nonmotor"), ('seizure_old', "atypicalabscence"), ('seizure_new', "GAS"),
                         ('onset', "generalized"), ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-06-22 07:47:28")), ('end', pd.Timestamp("2018-06-22 07:47:31")),
             ('motor', "nonmotor"), ('seizure_old', "atypicalabscence"), ('seizure_new', "GAS"),
             ('onset', "generalized"), ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-06-22 07:47:58")), ('end', pd.Timestamp("2018-06-22 07:48:00")),
             ('motor', "nonmotor"), ('seizure_old', "atypicalabscence"), ('seizure_new', "GAS"),
             ('onset', "generalized"), ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_145": [dict([('start', pd.Timestamp("2018-06-23 11:26:01")), ('end', pd.Timestamp("2018-06-23 11:26:25")),
                         ('motor', "nonmotor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
                         ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2018-06-23 12:05:57")), ('end', pd.Timestamp("2018-06-23 12:06:26")),
             ('motor', "nonmotor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2018-06-23 11:32:25")), ('end', pd.Timestamp("2018-06-23 11:32:35")),
             ('motor', "nonmotor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2018-06-23 14:53:36")), ('end', pd.Timestamp("2018-06-23 14:53:46")),
             ('motor', "nonmotor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-06-22 15:10:47")), ('end', pd.Timestamp("2018-06-22 15:11:44")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2018-06-22 16:32:49")), ('end', pd.Timestamp("2018-06-22 16:34:49")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-06-23 00:17:53")), ('end', pd.Timestamp("2018-06-23 00:18:11")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2018-06-23 11:40:29")), ('end', pd.Timestamp("2018-06-23 11:42:04")),
             ('motor', "bilateraltonicclonic"), ('seizure_old', "GTCS"), ('seizure_new', "TCS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Excel")])],
        "BN_146": [dict([('start', pd.Timestamp("2018-07-02 04:40:14")), ('end', pd.Timestamp("2018-07-02 04:42:14")),
                         ('motor', "bilateraltonicclonic"), ('seizure_old', "GTCS"), ('seizure_new', "TCS"),
                         ('onset', "temporal"), ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_147": [],
        "BN_149": [dict([('start', pd.Timestamp("2018-07-12 16:18:44")), ('end', pd.Timestamp("2018-07-12 16:19:30")),
                         ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "frontal"),
                         ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-07-12 17:23:11")), ('end', pd.Timestamp("2018-07-12 17:23:45")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "frontal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-07-12 18:20:23")), ('end', pd.Timestamp("2018-07-12 18:20:57")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "frontal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-07-12 21:32:34")), ('end', pd.Timestamp("2018-07-12 21:33:21")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "frontal"),
             ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2018-07-12 22:06:27")), ('end', pd.Timestamp("2018-07-12 22:07:04")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "frontal"),
             ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2018-07-12 22:51:14")), ('end', pd.Timestamp("2018-07-12 22:51:46")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "frontal"),
             ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2018-07-12 23:41:21")), ('end', pd.Timestamp("2018-07-12 23:41:51")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "frontal"),
             ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2018-07-13 00:15:17")), ('end', pd.Timestamp("2018-07-13 00:16:00")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "frontal"),
             ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2018-07-13 01:59:31")), ('end', pd.Timestamp("2018-07-13 02:00:00")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "frontal"),
             ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2018-07-13 02:26:17")), ('end', pd.Timestamp("2018-07-13 02:26:57")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "frontal"),
             ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2018-07-13 03:00:26")), ('end', pd.Timestamp("2018-07-13 03:01:12")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "frontal"),
             ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2018-07-13 03:39:44")), ('end', pd.Timestamp("2018-07-13 03:40:17")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "frontal"),
             ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2018-07-13 04:00:21")), ('end', pd.Timestamp("2018-07-13 04:01:07")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "frontal"),
             ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2018-07-13 04:23:24")), ('end', pd.Timestamp("2018-07-13 04:24:07")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "frontal"),
             ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2018-07-13 04:50:10")), ('end', pd.Timestamp("2018-07-13 04:50:56")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "frontal"),
             ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2018-07-13 05:32:04")), ('end', pd.Timestamp("2018-07-13 05:32:41")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "frontal"),
             ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2018-07-13 06:13:10")), ('end', pd.Timestamp("2018-07-13 06:13:50")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "frontal"),
             ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2018-07-13 10:03:04")), ('end', pd.Timestamp("2018-07-13 10:03:36")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "frontal"),
             ('awareness', "aware"), ('source', "Both")])],
        "BN_150": [],
        "BN_152": [],
        "BN_154": [],
        "BN_158": [],
        "BN_159": [dict([('start', pd.Timestamp("2018-08-19 01:26:13")), ('end', pd.Timestamp("2018-08-19 01:26:43")),
                         ('motor', "nonmotor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
                         ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2018-08-19 07:49:07")), ('end', pd.Timestamp("2018-08-19 07:49:45")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2018-08-20 17:42:29")), ('end', pd.Timestamp("2018-08-20 17:44:01")),
             ('motor', "nonmotor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-08-21 06:12:52")), ('end', pd.Timestamp("2018-08-21 06:15:09")),
             ('motor', "nonmotor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2018-08-21 08:25:33")), ('end', pd.Timestamp("2018-08-21 08:28:30")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2018-08-17 06:05:10")), ('end', pd.Timestamp("2018-08-17 06:06:07")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-08-18 07:56:10")), ('end', pd.Timestamp("2018-08-18 07:57:59")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2018-08-20 07:43:02")), ('end', pd.Timestamp("2018-08-20 07:45:50")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2018-08-21 11:35:03")), ('end', pd.Timestamp("2018-08-21 11:35:40")),
             ('motor', "nonmotor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")])],
        "BN_160": [dict([('start', pd.Timestamp("2018-08-22 15:46:40")), ('end', pd.Timestamp("2018-08-22 15:48:08")),
                         ('motor', "nonmotor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
                         ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-08-24 09:51:43")), ('end', pd.Timestamp("2018-08-24 09:52:03")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-08-24 09:53:51")), ('end', pd.Timestamp("2018-08-24 09:54:18")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-08-24 09:55:47")), ('end', pd.Timestamp("2018-08-24 09:56:08")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-08-24 10:00:26")), ('end', pd.Timestamp("2018-08-24 10:00:53")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")])],
        "BN_161": [],
        "BN_165": [dict([('start', pd.Timestamp("2018-10-10 10:56:22")), ('end', pd.Timestamp("2018-10-10 10:58:01")),
                         ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
                         ('awareness', "impairedawareness"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2018-10-14 10:54:42")), ('end', pd.Timestamp("2018-10-14 10:57:56")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2018-10-15 07:36:09")), ('end', pd.Timestamp("2018-10-15 07:37:35")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Excel")])],
        "BN_166": [dict([('start', pd.Timestamp("2018-10-15 16:08:20")), ('end', pd.Timestamp("2018-10-15 16:09:52")),
                         ('motor', "bilateraltonicclonic"), ('seizure_old', "GTCS"), ('seizure_new', "TCS"),
                         ('onset', "temporal"), ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_167": [dict([('start', pd.Timestamp("2018-10-19 05:56:16")), ('end', pd.Timestamp("2018-10-19 05:56:58")),
                         ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "unknown"),
                         ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-10-19 05:08:24")), ('end', pd.Timestamp("2018-10-19 05:09:39")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "unknown"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-10-19 06:17:04")), ('end', pd.Timestamp("2018-10-19 06:17:13")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "unknown"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-10-19 06:29:09")), ('end', pd.Timestamp("2018-10-19 06:29:56")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "unknown"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-10-19 06:31:32")), ('end', pd.Timestamp("2018-10-19 06:32:03")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "unknown"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-10-19 06:36:43")), ('end', pd.Timestamp("2018-10-19 06:37:15")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "unknown"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-10-19 06:38:53")), ('end', pd.Timestamp("2018-10-19 06:39:34")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "unknown"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-10-19 06:46:56")), ('end', pd.Timestamp("2018-10-19 06:47:19")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "unknown"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-10-19 06:49:23")), ('end', pd.Timestamp("2018-10-19 06:49:59")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "unknown"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-10-19 08:31:35")), ('end', pd.Timestamp("2018-10-19 08:32:02")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "unknown"),
             ('awareness', "impairedawareness"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2018-10-19 08:40:33")), ('end', pd.Timestamp("2018-10-19 08:41:15")),
             ('motor', "nonmotor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "unknown"),
             ('awareness', "aware"), ('source', "Excel")])],
        "BN_168": [dict([('start', pd.Timestamp("2018-10-26 15:03:29")), ('end', pd.Timestamp("2018-10-26 15:03:50")),
                         ('motor', "nonmotor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
                         ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-10-27 13:40:51")), ('end', pd.Timestamp("2018-10-27 13:41:51")),
             ('motor', "nonmotor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2018-10-27 14:31:17")), ('end', pd.Timestamp("2018-10-27 14:32:06")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Excel")])],
        "BN_169": [dict([('start', pd.Timestamp("2018-11-09 12:46:59")), ('end', pd.Timestamp("2018-11-09 12:47:34")),
                         ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
                         ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-11-09 13:35:25")), ('end', pd.Timestamp("2018-11-09 13:36:07")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")])],
        "BN_170": [dict([('start', pd.Timestamp("2018-11-13 17:07:15")), ('end', pd.Timestamp("2018-11-13 17:08:38")),
                         ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
                         ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-11-13 21:55:53")), ('end', pd.Timestamp("2018-11-13 21:57:59")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-11-14 07:32:05")), ('end', pd.Timestamp("2018-11-14 07:35:53")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_171": [dict([('start', pd.Timestamp("2018-11-27 11:12:05")), ('end', pd.Timestamp("2018-11-27 11:17:22")),
                         ('motor', "psychogenic"), ('seizure_old', "PNES"), ('seizure_new', "unknown"),
                         ('onset', "unknown"), ('awareness', "unknown"), ('source', "Excel")])],
        "BN_173": [dict([('start', pd.Timestamp("2018-12-19 18:57:38")), ('end', pd.Timestamp("2018-12-19 19:00:23")),
                         ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
                         ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2018-12-20 08:32:33")), ('end', pd.Timestamp("2018-12-20 08:34:03")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2018-12-21 06:59:07")), ('end', pd.Timestamp("2018-12-21 07:00:10")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Excel")])],
        "BN_176": [],
        "BN_177": [dict([('start', pd.Timestamp("2019-01-26 13:35:45")), ('end', pd.Timestamp("2019-01-26 13:36:44")),
                         ('motor', "nonmotor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
                         ('awareness', "nonaware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2019-01-26 15:03:24")), ('end', pd.Timestamp("2019-01-26 15:04:41")),
             ('motor', "nonmotor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "nonaware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2019-01-26 18:58:46")), ('end', pd.Timestamp("2019-01-26 19:00:44")),
             ('motor', "nonmotor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "nonaware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2019-01-26 21:30:00")), ('end', pd.Timestamp("2019-01-26 21:36:00")),
             ('motor', "nonmotor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "nonaware"), ('source', "Excel")])],
        "BN_178": [dict([('start', pd.Timestamp("2019-01-25 14:34:10")), ('end', pd.Timestamp("2019-01-25 14:35:33")),
                         ('motor', "nonmotor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
                         ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2019-01-26 18:02:42")), ('end', pd.Timestamp("2019-01-26 18:03:23")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2019-01-26 21:59:49")), ('end', pd.Timestamp("2019-01-26 22:01:13")),
             ('motor', "nonmotor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_179": [dict([('start', pd.Timestamp("2019-02-07 08:51:33")), ('end', pd.Timestamp("2019-02-07 08:53:24")),
                         ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
                         ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2019-02-04 19:40:11")), ('end', pd.Timestamp("2019-02-04 19:52:37")),
             ('motor', "bilateraltonicclonic"), ('seizure_old', "GTCS"), ('seizure_new', "TCS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_180": [dict([('start', pd.Timestamp("2019-02-16 00:22:00")), ('end', pd.Timestamp("2019-02-16 00:22:52")),
                         ('motor', "nonmotor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
                         ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2019-02-16 02:26:10")), ('end', pd.Timestamp("2019-02-16 02:28:40")),
             ('motor', "nonmotor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2018-02-15 18:12:46")), ('end', pd.Timestamp("2018-02-15 18:13:24")),
             ('motor', "nonmotor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2019-02-17 02:02:31")), ('end', pd.Timestamp("2019-02-17 02:04:46")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2019-02-17 08:01:11")), ('end', pd.Timestamp("2019-02-17 08:02:25")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2019-02-20 02:50:30")), ('end', pd.Timestamp("2019-02-20 02:51:53")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2019-02-20 07:31:02")), ('end', pd.Timestamp("2019-02-20 07:32:45")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_186": [dict([('start', pd.Timestamp("2019-04-14 10:48:00")), ('end', pd.Timestamp("2019-04-14 10:49:29")),
                         ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
                         ('awareness', "impairedawareness"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2019-04-14 22:03:27")), ('end', pd.Timestamp("2019-04-14 22:05:58")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2019-04-15 04:55:05")), ('end', pd.Timestamp("2019-04-15 04:57:21")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2019-04-15 06:46:48")), ('end', pd.Timestamp("2019-04-15 06:47:56")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Both")])],
        "BN_187": [],
        "BN_189": [dict([('start', pd.Timestamp("2019-04-19 02:34:29")), ('end', pd.Timestamp("2019-04-19 02:35:40")),
                         ('motor', "bilateraltonicclonic"), ('seizure_old', "GTCS"), ('seizure_new', "TCS"),
                         ('onset', "generalized"), ('awareness', "impairedawareness"), ('source', "Excel")])],
        "BN_193": [dict([('start', pd.Timestamp("2019-05-15 21:00:36")), ('end', pd.Timestamp("2019-05-15 21:01:01")),
                         ('motor', "nonmotor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
                         ('awareness', "impairedawareness"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2019-05-16 03:12:02")), ('end', pd.Timestamp("2019-05-16 03:12:20")),
             ('motor', "unclassified"), ('seizure_old', "unclassified/subclinical"), ('seizure_new', "unclassified"),
             ('onset', "temporal"), ('awareness', "unclassified"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2019-05-16 03:13:35")), ('end', pd.Timestamp("2019-05-16 03:14:32")),
             ('motor', "unclassified"), ('seizure_old', "unclassified/subclinical"), ('seizure_new', "unclassified"),
             ('onset', "temporal"), ('awareness', "unclassified"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2019-05-16 05:30:37")), ('end', pd.Timestamp("2019-05-16 05:31:08")),
             ('motor', "unclassified"), ('seizure_old', "unclassified/subclinical"), ('seizure_new', "unclassified"),
             ('onset', "temporal"), ('awareness', "unclassified"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2019-05-16 07:01:56")), ('end', pd.Timestamp("2019-05-16 07:02:55")),
             ('motor', "unclassified"), ('seizure_old', "unclassified/subclinical"), ('seizure_new', "unclassified"),
             ('onset', "temporal"), ('awareness', "unclassified"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2019-05-15 17:08:50")), ('end', pd.Timestamp("2019-05-15 17:09:36")),
             ('motor', "nonmotor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2019-05-15 18:15:42")), ('end', pd.Timestamp("2019-05-15 18:16:28")),
             ('motor', "nonmotor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2019-05-15 18:46:51")), ('end', pd.Timestamp("2019-05-15 18:47:31")),
             ('motor', "nonmotor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2019-05-15 22:27:51")), ('end', pd.Timestamp("2019-05-15 22:28:28")),
             ('motor', "nonmotor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2019-05-15 19:23:47")), ('end', pd.Timestamp("2019-05-15 19:24:31")),
             ('motor', "nonmotor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2019-05-16 01:57:00")), ('end', pd.Timestamp("2019-05-16 01:58:21")),
             ('motor', "nonmotor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2019-05-15 22:03:04")), ('end', pd.Timestamp("2019-05-15 22:03:20")),
             ('motor', "unclassified"), ('seizure_old', "unclassified/subclinical"), ('seizure_new', "unclassified"),
             ('onset', "temporal"), ('awareness', "unclassified"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2019-05-15 23:33:22")), ('end', pd.Timestamp("2019-05-15 23:34:00")),
             ('motor', "unclassified"), ('seizure_old', "unclassified/subclinical"), ('seizure_new', "unclassified"),
             ('onset', "temporal"), ('awareness', "unclassified"), ('source', "Both")])],
        "BN_195": [dict([('start', pd.Timestamp("2019-06-03 13:52:25")), ('end', pd.Timestamp("2019-06-03 13:54:03")),
                         ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
                         ('awareness', "impairedawareness"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2019-06-03 21:57:15")), ('end', pd.Timestamp("2019-06-03 21:59:04")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2019-06-03 22:33:42")), ('end', pd.Timestamp("2019-06-03 22:34:30")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2019-06-03 03:05:33")), ('end', pd.Timestamp("2019-06-03 03:07:00")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Excel")])],
        "BN_197": [dict([('start', pd.Timestamp("2019-06-14 08:02:25")), ('end', pd.Timestamp("2019-06-14 08:04:05")),
                         ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
                         ('awareness', "impairedawareness"), ('source', "Excel")])],
        "BN_198": [dict([('start', pd.Timestamp("2019-06-25 05:41:58")), ('end', pd.Timestamp("2019-06-25 05:42:24")),
                         ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "parietal"),
                         ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2019-06-25 06:34:34")), ('end', pd.Timestamp("2019-06-25 06:35:05")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "parietal"),
             ('awareness', "aware"), ('source', "Excel")]), dict(
            [('start', pd.Timestamp("2019-06-25 07:17:34")), ('end', pd.Timestamp("2019-06-25 07:18:03")),
             ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "parietal"),
             ('awareness', "aware"), ('source', "Excel")])],
        "BN_199": [dict([('start', pd.Timestamp("2019-07-07 15:36:06")), ('end', pd.Timestamp("2019-07-07 15:37:41")),
                         ('motor', "motor"), ('seizure_old', "SPS"), ('seizure_new', "FAS"), ('onset', "temporal"),
                         ('awareness', "aware"), ('source', "Both")]), dict(
            [('start', pd.Timestamp("2019-07-07 01:56:46")), ('end', pd.Timestamp("2019-07-07 01:59:01")),
             ('motor', "motor"), ('seizure_old', "CPS"), ('seizure_new', "FIAS"), ('onset', "temporal"),
             ('awareness', "impairedawareness"), ('source', "Excel")])]
    }
    return all

if __name__ == "__main__":
    all_seizures = get_all_seizures()
    only_motor = get_motor_seizures()
    distribution_of_seizures_of_kind_and_motor = get_num_of_seizure_and_motor(all_seizures)
    num_of_patients_of_seizure_and_motor = get_num_of_seizure_and_motor_per_patient(all_seizures)
    patients_with_seizure_kind = get_num_of_patients_with_seizure_kind(all_seizures)
    print(num_of_patients_of_seizure_and_motor)