def make_dics(connections):
    fwd = {}
    rev = {}
    for c in connections:
        i = c[0]
        o = c[1]
        try:
            fwd[i].append(o)
        except KeyError:
            fwd[i] = [o]
        try:
            rev[o].append(i)
        except KeyError:
            rev[o] = [i]
    return fwd, rev


def make_network(gen: Genome):
    # FInd what's input and output
    fwd, rev = make_dics(gen.active_connects.keys())
    outputs = [n for n in gen.nodes if n not in fwd.keys()]
    inputs = [n for n in gen.nodes if n not in rev.keys()]
    
    # Record what goes into each layer, the matrix, and what comes out of each layer
    layers = []
    in_ls = []
    out_ls = []
    nn_layers = []
    max_node = 0
    
    # Go backwards and determine the network graph
    nodes_to_process = deepcopy(outputs)
    while nodes_to_process:
        next_layer = []
        for n in nodes_to_process:
            if n > max_node:
                max_node = n
            try:
                next_layer.extend(rev[n])
            except KeyError:
                pass
        layers.append(list(set(next_layer)))
        nodes_to_process = next_layer
    layers.reverse()
    layers = layers[1:]
    layers.append(outputs)
    # Build the model
    if len(layers) > 1:
        for i in range(len(layers) - 1):
            if i == 0:
                in_l = inputs
            else:
                in_l = layers[i]
            if i == len(layers) - 1:
                out_l = outputs
            else:
                out_l = layers[i+1]
            out_l = [n for n in out_l if n not in inputs]
            nn_layer = nn.Linear(len(in_l), len(out_l), bias=False)
            for i, ni in enumerate(in_l):
                for j, no in enumerate(out_l):
                    if ni == no:
                        w = 1
                    else:
                        try:
                            w = gen.active_connects[(ni, no)]
                        except KeyError:
                            w = 0
                    with torch.no_grad():
                        nn_layer.weight[j, i] = w
            nn_layers.append(nn_layer)
            in_ls.append(in_l)
            out_ls.append(out_l)
    else:
        out_l = layers[0]
    
    def model(val_in):
        if len(val_in.shape) == 1:
            val_in = torch.reshape(val_in, [1, -1])
        assert len(val_in.shape) == 2, 'Requires 2D Tensor'
        values = torch.zeros(val_in.shape[0], max_node+1)
        values[:, inputs] = val_in
        for i, layer in enumerate(nn_layers):
            ins = values[:, in_ls[i]]
            loop_value = layer(ins)
            if i != len(nn_layers) - 1:
                pass
            loop_value = torch.sigmoid(loop_value)
            values[:, out_ls[i]] = loop_value
        
        return values[:, outputs]
    
    return model
