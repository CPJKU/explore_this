import matplotlib.pyplot as plt
import torch

from explore_this.metrics import mask_recall
import explore_this.architecture_testing.listen as listen
from explore_this.model.grid import GridOutput, WindowedGrid
from explore_this.model.subgrid import BeatSubgrid

from tqdm import tqdm

from explore_this.model.beat_tracker import ExploreThis, ModelOutput
from explore_this.model.grid_space_embedding import get_state


def plot_bins(
    features: torch.Tensor,
    grid_processor: WindowedGrid,
    truth,
    width=300,
    loss_handler=None,
):
    """
    Feature should not have the batch dimension
    loss_handler should be a function that takes a GridOutput and a truth mask, and returns a list of strings.

    example:
        def loss_handler(output, truth):
        reg = model.grid_reg_loss(output.features)
        pred = model.grid_recall_loss(output.activation, truth)
        total = reg + pred
        print(f"Total: {total:.4f}")
        print("Regularization:", reg)
        print("Prediction:", pred)


        plot_bins(
            features=output.features[0],
            grid_processor=gp,
            truth=batch["truth_beat"][0],
            loss_handler=loss_handler,
        )
    """

    base_probs = grid_processor.probabilities(features)
    # print(base_probs.shape)

    for i_fbin in range(grid_processor.n_bins):
        tmp_feats = features.clone().detach()
        tmp_feats[..., 0] = 0
        tmp_feats[..., i_fbin, 0] = 1000

        output = grid_processor(tmp_feats.unsqueeze(0)).squeeze(0)
        activation = output.activation

        plt.plot(activation[:width], label="Activation")
        plt.plot(truth[:width], label="Truth")

        base_prob = base_probs[..., i_fbin].mean()
        plt.title(f"Bin {i_fbin} with mean probability {base_prob:.4%}")
        plt.legend()
        plt.show()

        if loss_handler is not None:
            loss_handler(output, truth)


def compare_annotations_and_predictions(
    batch,
    output: ModelOutput,
    idx,
    xlim: None | tuple[float, float] = None,
    audio=True,
    full_piece: bool = False,
):
    print(batch["spect_path"][idx])
    if audio:
        print("Truth")
        listen.listen_to_annotations(
            batch["spect_path"][idx], start_frame=batch["start_frame"][idx].item()
        )

        print("Prediction")
        listen.listen_to_model_output(
            batch["spect_path"][idx],
            output[idx],
            batch["start_frame"][idx] if not full_piece else None,
        )

    b_indeces = torch.nonzero(batch["truth_beat"][idx], as_tuple=True)[0]
    db_indeces = torch.nonzero(batch["truth_downbeat"][idx], as_tuple=True)[0]

    plt.scatter(
        db_indeces,
        torch.ones_like(db_indeces) * 1.35,
        marker="|",
        label="true downbeats",
    )
    plt.scatter(
        b_indeces, torch.ones_like(b_indeces) * 1.3, marker="|", label="true beats"
    )

    dbm_indeces = torch.nonzero(output[idx].downbeat_mask.squeeze(0), as_tuple=True)[0]
    bm_indeces = torch.nonzero(output[idx].beat_mask.squeeze(0), as_tuple=True)[0]
    gm_indeces = torch.nonzero(output[idx].grid_mask.squeeze(0), as_tuple=True)[0]

    plt.scatter(
        dbm_indeces,
        torch.ones_like(dbm_indeces) * 1.2,
        marker="|",
        label="predicted downbeats",
    )
    plt.scatter(
        bm_indeces,
        torch.ones_like(bm_indeces) * 1.15,
        marker="|",
        label="predicted beats",
    )
    plt.scatter(
        gm_indeces,
        torch.ones_like(gm_indeces) * 1.1,
        marker="|",
        label="predicted grid",
    )

    plt.plot(
        output[idx].beat_activation.detach().squeeze(0),
        alpha=0.8,
        label="Beat probability",
    )
    plt.plot(
        -output[idx].downbeat_activation.detach().squeeze(0),
        alpha=0.8,
        label="Downbeat probability",
    )

    if xlim is not None:
        plt.xlim(*xlim)

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    plt.show()


def compare_annotations_and_grid_predictions(
    batch, output: GridOutput, idx, width=1500, audio=True, full_piece: bool = False
):
    print(batch["spect_path"][idx])
    print("Recall: ")
    print(mask_recall(output.mask, batch["truth_beat"])[idx].item())

    if audio:
        print("Truth")
        listen.listen_to_annotations(
            batch["spect_path"][idx], start_frame=batch["start_frame"][idx].item()
        )
        print("Prediction")
        listen.listen_to_prediction(
            batch["spect_path"][idx],
            output.mask[idx],
            batch["start_frame"][idx] if not full_piece else None,
            metronome_volume=1,
        )

    plt.figure(figsize=(15, 5))
    plt.plot(output.mask[idx][:width], alpha=0.5, label="Predicted mask")
    plt.plot(output.activation[idx].detach()[:width], alpha=0.5, label="Activation")
    plt.plot(batch["truth_beat"][idx][:width], alpha=0.5, label="Truth")
    plt.legend()
    plt.show()


def compare_models(
    a: ExploreThis, b: ExploreThis, dataloader, n_batches=8
) -> list[tuple]:
    evaluations = []

    for i, batch in tqdm(enumerate(dataloader)):
        if i == n_batches:
            break

        output_a = a.forward(batch["spect"].float())
        rec_a = mask_recall(output_a.grid_mask, batch["truth_beat"])

        output_b = b.forward(batch["spect"].float())
        rec_b = mask_recall(output_b.grid_mask, batch["truth_beat"])

        evaluations += [
            (
                (rec_b[j] - rec_a[j]).abs(),
                output_a[j],
                output_b[j],
                rec_a[j],
                rec_b[j],
                batch,
                j,
            )
            for j in range(len(rec_a))
        ]

    evaluations.sort(key=lambda x: x[0], reverse=True)
    return evaluations


def show_comparison(evaluations, n_top=5, max_width=500, audio=True, full_piece=False):
    """Shows the result of compare_models."""
    for i, (_, oa, ob, ra, rb, batch, idx) in enumerate(evaluations[:n_top]):
        print(f"{i}: {batch['spect_path'][idx]}")
        plt.figure(figsize=(15, 5))
        plt.plot(batch["truth_beat"][idx][:max_width], label="Truth")
        plt.plot(
            oa.activation[0].detach()[:max_width], alpha=0.5, label=f"Model A: {ra:.3f}"
        )
        plt.plot(
            ob.activation[0].detach()[:max_width], alpha=0.5, label=f"Model B: {rb:.3f}"
        )
        plt.legend()
        plt.show()

        if audio:
            print("Truth")
            listen.listen_to_annotations(
                batch["spect_path"][idx], start_frame=batch["start_frame"][idx].item()
            )
            print("Prediction A")
            listen.listen_to_prediction(
                batch["spect_path"][idx],
                oa.mask[0],
                batch["start_frame"][idx] if not full_piece else None,
                metronome_volume=1,
            )
            print("Prediction B")
            listen.listen_to_prediction(
                batch["spect_path"][idx],
                ob.mask[0],
                batch["start_frame"][idx] if not full_piece else None,
                metronome_volume=1,
            )


def visualize_meter_phase_dist(
    features: torch.Tensor,
    sg: BeatSubgrid,
    prob_threshold=0.01,
    unroll=False,
):
    # Assume x is (T, C), T is grid index and C is feature dimension
    Y = features.softmax(dim=-1)
    if unroll:
        indices = torch.arange(Y.shape[1])
        for t in range(Y.shape[0]):
            Y[t, :] = Y[t, indices]
            indices = sg.next_index[indices]

    probability_per_state = features.softmax(dim=-1).mean(dim=0)

    # Create classes of buckets with the same beat/downbeat meter

    combined_states = [sg.split_combined_state(i) for i in range(sg.input_size)]

    combined_states = torch.Tensor(list(zip(*combined_states)))
    bstates = get_state(combined_states[0])
    dbstates = get_state(combined_states[1])

    masks = {}
    for b in range(sg.max_beat_meter):
        for d in range(sg.max_downbeat_meter):
            mask = (bstates[:, 0] == b + 1) & (dbstates[:, 0] == d + 1)
            masks[(b + 1, d + 1)] = mask

    classes = list(masks.keys())
    classprobs = torch.zeros(len(classes))

    for i, mask_idx in enumerate(classes):
        mask = masks[mask_idx]
        classprobs[i] = probability_per_state[mask].sum()

    order = torch.argsort(classprobs, descending=True)
    classes = [classes[o] for o in order]
    classprobs = classprobs[order]

    for i, c in enumerate(classes):
        if classprobs[i] < prob_threshold:
            break
        print(f"beat : {c[0]} grid - downbeat : {c[1]} beat")
        print(f"Probability {classprobs[i]:.4f}")
        plt.imshow(Y[:, masks[c]], aspect="auto", vmin=0, vmax=1)
        plt.show()
