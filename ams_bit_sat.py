var_names = ["ix", "iy", "iz", "xi", "xx", "xy", "xz", "yi", "yx", "yy", "yz", "zi", "zx", "zy", "zz"]
equations = [
    ("ix", "xi", "xx", 0),
    ("iy", "xi", "xy", 0),
    ("iz", "xi", "xz", 0),
    ("ix", "yi", "yx", 0),
    ("iy", "yi", "yy", 0),
    ("iz", "yi", "yz", 0),
    ("ix", "zi", "zx", 0),
    ("iy", "zi", "zy", 0),
    ("iz", "zi", "zz", 0),
    ("yz", "zy", "xx", 0),
    ("zx", "xz", "yy", 0),
    ("xy", "yx", "zz", 0),
    ("xx", "yy", "zz", 1),
    ("xy", "yz", "zx", 1),
    ("xz", "yx", "zy", 1),
]

max_sat = 0

for j in range(1 << len(var_names)):
    vals = {}
    for k, name in enumerate(var_names):
        vals[name] = (j >> k) % 2
    sat = 0
    # check satisfaction of each equation
    for var1, var2, var3, result in equations:
        # if bit assignment satisfies, add to number of satisfied equations
        if (vals[var1] + vals[var2] + vals[var3]) % 2 == result:
            sat += 1
    # check if new max
    if sat > max_sat:
        max_sat = sat

print("Maximum number of satisfied equations: ", max_sat)
