#!/usr/bin/env python3

import argparse
import os
import shutil
import sys

import matplotlib        as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import radical.analytics as ra
import radical.pilot     as rp
import radical.utils     as ru

PAGE_WIDTH   = 516
CORRECTION   = 0.5

plt.style.use(ra.get_mplstyle('radical_mpl'))
mpl.rcParams['text.usetex'] = False
mpl.rcParams['font.serif']  = ['Nimbus Roman Becker No9L']
mpl.rcParams['font.family'] = 'serif'

DURATIONS = {
    'boot'        : [{ru.EVENT: 'bootstrap_0_start'},
                     {ru.EVENT: 'bootstrap_0_ok'   }],
    'agent_setup' : [{ru.EVENT: 'bootstrap_0_ok'   },
                     {ru.STATE: rp.PMGR_ACTIVE     }],
    'exec_queue'  : [{ru.EVENT: 'schedule_ok'      },
                     {ru.STATE: rp.AGENT_EXECUTING }],
    'exec_prep'   : [{ru.STATE: rp.AGENT_EXECUTING },
                     {ru.EVENT: 'task_run_start'   }],
    'exec_rp'     : [{ru.EVENT: 'task_run_start'   },
                     {ru.EVENT: 'launch_start'     }],
    'exec_sh'     : [{ru.EVENT: 'launch_start'     },
                     {ru.EVENT: 'launch_submit'    }],
    'exec_launch' : [{ru.EVENT: 'launch_submit'    },
                     {ru.EVENT: 'exec_start'       }],
    'exec_cmd'    : [{ru.EVENT: 'exec_start'       },
                     {ru.EVENT: 'exec_stop'        }],
    'exec_finish' : [{ru.EVENT: 'exec_stop'        },
                     {ru.EVENT: 'launch_collect'   }],
    'term_sh'     : [{ru.EVENT: 'launch_collect'   },
                     {ru.EVENT: 'launch_stop'      }],
    'term_rp'     : [{ru.EVENT: 'launch_stop'      },
                     {ru.EVENT: 'task_run_stop'    }],
    'unschedule'  : [{ru.EVENT: 'task_run_stop'    },
                     {ru.EVENT: 'unschedule_stop'  }]
}

METRICS = [
    ['Bootstrap',  ['boot', 'agent_setup'],        '#c6dbef'],
#    ['Warmup',     ['warm'],                       '#f0f0f0'],
    ['Schedule',   ['exec_queue', 'unschedule'],   '#c994c7'],
#    ['Exec setup', ['exec_prep', 'exec_rp', 'exec_sh',
#                    'term_sh', 'term_rp'],         '#fdbb84'],
#    ['Launching',  ['exec_launch', 'exec_finish'], '#ff9999'],
    ['Running',    ['exec_cmd'],                   '#e31a1c'],
#    ['Cooldown',   ['drain'],                      '#addd8e']
]

# ------------------------------------------------------------------------------


class Plotter:

    def __init__(self, input_dir, plots_dir, sid=None):

        self.input_dir = input_dir
        self.plots_dir = plots_dir

        self.sid  = sid
        self.data = {}

    def load_session(self, sid=None):

        if self.sid and self.data:
            if not sid or sid == self.sid:
                return

        self.sid = sid or self.sid
        if not self.sid:
            raise RuntimeError('Session ID not provided')

        session = ra.Session(
            os.path.join(self.input_dir, self.sid), 'radical.pilot')

        self.data.update({
            'session': session,
            'pilots' : session.filter(etype='pilot', inplace=False),
            'tasks'  : session.filter(etype='task',  inplace=False)})
        self.data.update({
            'pid': self.data['pilots'].list('uid')[0]})

    def plot_utilization(self, sid=None, x_limits=None):

        self.load_session(sid=sid)

        sid = self.sid
        pid = self.data['pid']

        threads_per_core = int(os.environ.get('RADICAL_SMT', 1))
        rtype_info = {
            'cpu': {'label': 'Number of CPU cores',
                    'formatter': lambda z, pos: int(z / threads_per_core)},
            'gpu': {'label': 'Number of GPUs',
                    'formatter': None}
        }

        exp = ra.Experiment(
            [os.path.join(self.input_dir, self.sid)], stype='radical.pilot')
        # get the start time of each pilot
        p_zeros = ra.get_pilots_zeros(exp)

        fig, axarr = plt.subplots(1, 2, figsize=(
            ra.get_plotsize(PAGE_WIDTH, subplots=(1, 2))))

        sub_label = 'a'
        legend = None
        for idx, rtype in enumerate(rtype_info):

            consumed = rp.utils.get_consumed_resources(
                exp._sessions[0], rtype, {'consume': DURATIONS})

            # generate the subplot with labels
            legend, patches, x, y = ra.get_plot_utilization(
                METRICS, {sid: consumed}, p_zeros[sid][pid], sid)

            # place all the patches, one for each metric, on the axes
            for patch in patches:
                patch.set_y(patch.get_y() + CORRECTION)
                axarr[idx].add_patch(patch)

            if x_limits and isinstance(x_limits, (list, tuple)):
                axarr[idx].set_xlim(x_limits)
            else:
                axarr[idx].set_xlim([x['min'], x['max']])
            axarr[idx].set_ylim([int(y['min'] + CORRECTION),
                                 int(y['max'] + CORRECTION)])

            axarr[idx].xaxis.set_major_locator(mticker.MaxNLocator(5))
            axarr[idx].yaxis.set_major_locator(mticker.MaxNLocator(5))

            if rtype_info[rtype]['formatter'] is not None:
                axarr[idx].yaxis.set_major_formatter(mticker.FuncFormatter(
                    rtype_info[rtype]['formatter']))

            axarr[idx].set_xlabel('(%s)' % sub_label, labelpad=10)
            axarr[idx].set_ylabel(rtype_info[rtype]['label'])
            axarr[idx].set_title(' ')  # placeholder

            sub_label = chr(ord(sub_label) + 1)

        fig.legend(legend, [m[0] for m in METRICS],
                   loc='upper center',
                   bbox_to_anchor=(0.5, 1.03),
                   ncol=len(METRICS))
        fig.text(0.5, 0.05, 'Time (s)', ha='center')

        plt.tight_layout()
        plt.show()

        plot_name = '%s.utilization.png' % '.'.join(self.sid.rsplit('.', 2)[1:])
        fig.savefig(os.path.join(self.plots_dir, plot_name))

    def print_metrics(self, sid=None):

        self.load_session(sid=sid)

        for rtype in ['cpu', 'gpu']:
            _, _, stats_abs, stats_rel, info = self.data['session'].utilization(
                METRICS, rtype, {'consume': DURATIONS})
            print('- %s RU: ' % rtype.upper(), info)

    def plot_concurrency(self, sid=None):

        self.load_session(sid=sid)

        events = {'Tasks scheduling': [{ru.STATE: 'AGENT_SCHEDULING'},
                                       {ru.EVENT: 'schedule_ok'}],
                  'Tasks running'   : [{ru.EVENT: 'exec_start'},
                                       {ru.EVENT: 'exec_stop'}]}

        fig, ax = plt.subplots(figsize=ra.get_plotsize(PAGE_WIDTH))

        pilot_starttime = self.data['pilots'].\
            timestamps(event={ru.EVENT: 'bootstrap_0_start'})[0]

        time_series = {e_name: self.data['session'].
                       concurrency(event=events[e_name], sampling=1)
                       for e_name in events}

        for e_name in time_series:
            ax.plot([e[0] - pilot_starttime for e in time_series[e_name]],
                    [e[1] for e in time_series[e_name]],
                    label=ra.to_latex(e_name))

        fig.legend(['Tasks scheduling', 'Tasks running'],
                   loc='upper center',
                   bbox_to_anchor=(0.5, 1.0),
                   ncol=2)
        fig.text(0.01, 0.5, 'Number of tasks', va='center', rotation='vertical')
        fig.text(0.5, 0.01, 'Time (s)', ha='center')

        plt.tight_layout()
        plt.show()
        plot_name = '%s.concurrency.png' % '.'.join(self.sid.rsplit('.', 2)[1:])
        fig.savefig(os.path.join(self.plots_dir, plot_name))


# ------------------------------------------------------------------------------


def get_args():
    """
    Get arguments.
    :return: Arguments namespace.
    :rtype: _AttributeHolder
    """
    parser = argparse.ArgumentParser(
        description='Create plot of resource utilization for the EnTK app.',
        usage='entk-ru-plot.py --sid <session id> '
              '[--input_dir <sessions dir>'
              ' --plot_dir <plots dir>]')

    parser.add_argument(
        '--sid',
        dest='sid',
        type=str,
        help='session id',
        required=True)

    parser.add_argument(
        '--input_dir',
        dest='input_dir',
        type=str,
        help='directory with sessions',
        required=False,
        default='.')

    parser.add_argument(
        '--plot_dir',
        dest='plot_dir',
        type=str,
        help='directory for produced plots',
        required=False,
        default='.')

    return parser.parse_args(sys.argv[1:])


def proceed(args):
    """
    Proceed component execution.
    :param args: Arguments.
    :type args: _AttributeHolder
    """
    p = Plotter(input_dir=args.input_dir,
                plots_dir=args.input_dir,
                sid=args.sid)

    p.plot_utilization()
    p.print_metrics()
    p.plot_concurrency()


# ------------------------------------------------------------------------------
#
if __name__ == '__main__':

    # clean cache
    shutil.rmtree('%s/.radical/analytics/cache' % os.environ['HOME'],
                  ignore_errors=True)

    proceed(args=get_args())

# ------------------------------------------------------------------------------

