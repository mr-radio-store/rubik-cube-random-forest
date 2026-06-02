# -----------------------------
# Rubik's Cube 3 Sides with Random Forest Painter
# -----------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# 1️⃣ Define Google colors
# -----------------------------
colors_map = ['#4285F4', '#EA4335', '#FBBC05', '#34A853']  # Blue, Red, Yellow, Green

# -----------------------------
# 2️⃣ Prepare training data
# -----------------------------
# 3 sides, 3x3 stickers per side
X_train = []  # features: side, row, col
y_train = []  # labels: color index (0-3)

for side in range(3):
    for row in range(3):
        for col in range(3):
            X_train.append([side, row, col])
            y_train.append(np.random.randint(0, 4))  # randomly assign one of 4 colors

X_train = np.array(X_train)
y_train = np.array(y_train)

# -----------------------------
# 3️⃣ Train Random Forest
# -----------------------------
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# -----------------------------
# 4️⃣ Predict colors for each sticker
# -----------------------------
X_test = X_train
predicted_colors = clf.predict(X_test)
predicted_colors = predicted_colors.reshape(3, 3, 3)  # side, row, col

# -----------------------------
# 5️⃣ Plot 3 sides
# -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(6, 2))

for s in range(3):
    for r in range(3):
        for c in range(3):
            color_index = predicted_colors[s, r, c]
            axes[s].add_patch(plt.Rectangle((c, 2 - r), 1, 1, color=colors_map[color_index]))
    axes[s].set_xlim(0, 3)
    axes[s].set_ylim(0, 3)
    axes[s].axis('off')
    axes[s].set_title(f'Side {s + 1}')

plt.tight_layout()
plt.show()
