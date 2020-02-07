def index(s):
    _i_to = tuple(s)
    _to_i = {o: i for (i, o) in enumerate(_i_to)}
    return lambda o: _to_i[o], lambda i: _i_to[i]
