test_group_comparison():
    r0 = {
        'current_value': (0, 5),
        'conditional_sources': [(1, 1), (1, 2), (1, 3), (2, 1), (2, 0)],
        'conditional_target': [(0, 1), (0, 2), (0, 3)],
        'conditional_full': [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3),
                             (2, 1), (2, 0)],
        'omnibus_sign': True,
        'cond_sources_pval': np.array([0.001, 0.0014, 0.01, 0.045, 0.047])
        }
    r1 = {
        'current_value': (1, 5),
        'conditional_sources': [(2, 0), (2, 1), (2, 2), (3, 1), (3, 2)],
        'conditional_target': [(1, 0), (1, 1)],
        'conditional_full': [(1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (3, 1),
                             (3, 2)],
        'omnibus_sign': True,
        'cond_sources_pval': np.array([0.00001, 0.00014, 0.01, 0.035, 0.02])
        }
    r2 = {
        'current_value': (2, 5),
        'conditional_sources': [],
        'conditional_target': [(3, 0), (3, 1)],
        'conditional_full': [(3, 0), (3, 1)],
        'omnibus_sign': False,
        'cond_sources_pval': None
        }
    res_0 = {
        0: r0,
        1: r1,
        2: r2
    }

    r0 = {
        'current_value': (0, 5),
        'conditional_sources': [(2, 1), (2, 0)],
        'conditional_target': [(0, 1), (0, 2), (0, 4)],
        'conditional_full': [(0, 1), (0, 2), (0, 3), (2, 1), (2, 0)],
        'omnibus_sign': True,
        'cond_sources_pval': np.array([0.001, 0.0014, 0.01, 0.045, 0.047])
        }
    r1 = {
        'current_value': (1, 5),
        'conditional_sources': [(2, 0), (2, 1), (3, 2), (3, 0)],
        'conditional_target': [(1, 1), (1, 2), (1, 3)],
        'conditional_full': [(1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (3, 2),
                             (3, 0)],
        'omnibus_sign': True,
        'cond_sources_pval': np.array([0.00001, 0.00014, 0.01, 0.035, 0.02])
        }
    r3 = {
        'current_value': (3, 5),
        'conditional_sources': [(2, 0), (2, 1), (1, 0), (1, 1), (1, 2)],
        'conditional_target': [(3, 1), (3, 3)],
        'conditional_full': [(3, 1), (3, 3), (2, 0), (2, 1), (1, 0), (1, 1),
                             (1, 2)],
        'omnibus_sign': False,
        'cond_sources_pval': None
        }
    res_1 = {
        0: r0,
        1: r1,
        3: r3
    }

    data_1 = Data()
    data_1.generate_mute_data(100, 5)
    data_2 = Data()
    data_2.generate_mute_data(100, 5)
    options = {
        'cmi_calc_name': 'jidt_kraskov',
        'stats_type': 'independent',
        'n_perm_comp': 10,
        'alpha_comp': 0.5
        }

    comp = Compare_single_recording(options)
    comp.compare(res_0, res_1, data_1, data_2)

if __name__ == '__main__':
    test_group_comparison()
