import pandas as pd
import numpy as np

def qgl_weights(jet1, jet2, isHerwig, output, variables, njets):
    qgl = pd.DataFrame(index=output.index, columns=["wgt", "wgt_down"]).fillna(1.0)
    print(f"len qgl: {len(qgl)}")
    qgl1 = get_qgl_weights(jet1, isHerwig).fillna(1.0)
    qgl2 = get_qgl_weights(jet2, isHerwig).fillna(1.0)
    qgl.wgt *= qgl1 * qgl2
    # print(f"qgl1: {qgl1[output.event_selection]}")
    # print(f"qgl2: {qgl2[output.event_selection]}")
    # print(f"variables.njets: {variables.njets[output.event_selection]}")
    # print(f"len output.event_selection: {len(output.event_selection)}")
    # print(f"output.event_selection: {output.event_selection}")
    # print(f"jet1.pt: {jet1.pt[output.event_selection]}")
    # print(f"qgl.wgt: {qgl.wgt[output.event_selection]}")

    qgl.wgt[variables.njets == 1] = 1.0
    selected = output.event_selection & (njets > 2)
    print(f"qgl_mean: {qgl.wgt[selected].mean()}")
    print(f"qgl nom b4: {qgl.wgt[selected]}")
    qgl.wgt = qgl.wgt / qgl.wgt[selected].mean()
    
    print(f"qgl nom after: {qgl.wgt[selected]}")
    qgl = qgl.fillna(1.0)

    qgl_final = qgl.wgt[output.event_selection]
    # print(f"qgl nom final: {qgl_final.to_numpy()}")
    print(f"qgl nom final sum: {np.sum(qgl_final)}")

    wgts = {"nom": qgl.wgt, "up": qgl.wgt * qgl.wgt, "down": qgl.wgt_down}
    return wgts


def get_qgl_weights(jet, isHerwig):
    df = pd.DataFrame(index=jet.index, columns=["weights"])
    # df.fillna(1, inplace=True)
    # print(f"get_qgl_weights df: {df}")
    wgt_mask = (jet.partonFlavour != 0) & (abs(jet.eta) < 2) & (jet.qgl > 0)
    light = wgt_mask & (abs(jet.partonFlavour) < 4)
    gluon = wgt_mask & (jet.partonFlavour == 21)

    qgl = jet.qgl
    # print(f"get_qgl_weights qgl:{qgl}")
    # print(f"get_qgl_weights light:{light}")
    # print(f"get_qgl_weights gluon:{gluon}")
    # print(f"isHerwig:{isHerwig}")
    if isHerwig:
        df.weights[light] = (
            1.16636 * qgl[light] ** 3
            - 2.45101 * qgl[light] ** 2
            + 1.86096 * qgl[light]
            + 0.596896
        )
        df.weights[gluon] = (
            -63.2397 * qgl[gluon] ** 7
            + 111.455 * qgl[gluon] ** 6
            - 16.7487 * qgl[gluon] ** 5
            - 72.8429 * qgl[gluon] ** 4
            + 56.7714 * qgl[gluon] ** 3
            - 19.2979 * qgl[gluon] ** 2
            + 3.41825 * qgl[gluon]
            + 0.919838
        )
    else:
        df.weights[light] = (
            -0.666978 * qgl[light] ** 3
            + 0.929524 * qgl[light] ** 2
            - 0.255505 * qgl[light]
            + 0.981581
        )
        df.weights[gluon] = (
            -55.7067 * qgl[gluon] ** 7
            + 113.218 * qgl[gluon] ** 6
            - 21.1421 * qgl[gluon] ** 5
            - 99.927 * qgl[gluon] ** 4
            + 92.8668 * qgl[gluon] ** 3
            - 34.3663 * qgl[gluon] ** 2
            + 6.27 * qgl[gluon]
            + 0.612992
        )
    # print(f"df.weights: {df.weights}")
    return df.weights
