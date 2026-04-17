import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import config
from dataloader import to_5class


# ── Train / test split ────────────────────────────────────────────────────────

def split(X, Y):
    Y5 = to_5class(Y)
    return train_test_split(X, Y5,
                            test_size=config.TEST_SIZE,
                            random_state=config.RANDOM_SEED,
                            shuffle=True)


# ── Class balancing (undersample N, SMOTE for S/V/F/Q) ───────────────────────

def balance(X_train, y_train):
    idx0  = np.where(y_train == 0)[0]
    idx14 = np.where(y_train != 0)[0]

    # Undersample Normal class
    chosen = np.random.choice(idx0, config.N_UNDERSAMPLE, replace=False)
    X0, y0 = X_train[chosen], y_train[chosen]

    # SMOTE minority classes
    strategy = {1: config.SMOTE_TARGET, 2: config.SMOTE_TARGET,
                3: config.SMOTE_TARGET, 4: config.SMOTE_TARGET}
    sm = SMOTE(sampling_strategy=strategy, random_state=42)
    X14, y14 = sm.fit_resample(X_train[idx14], y_train[idx14])

    X_bal = np.vstack([X0, X14])
    y_bal = np.concatenate([y0, y14])

    perm = np.random.permutation(len(X_bal))
    return X_bal[perm], y_bal[perm]


# ── Final tensors ready for model.fit ─────────────────────────────────────────

def get_batches(X, Y):
    X_tr, X_te, y_tr, y_te = split(X, Y)

    # X_tr, y_tr = balance(X_tr, y_tr)

    # Add channel dimension expected by Conv1D
    X_tr = X_tr.reshape(-1, config.WINDOW_SIZE, 1).astype(np.float32)
    X_te = X_te.reshape(-1, config.WINDOW_SIZE, 1).astype(np.float32)

    return X_tr, X_te, y_tr, y_te
