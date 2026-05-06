#!/usr/bin/env python3
"""
Benchmark script for data-parallel BioReason-Pro on trn2.3xlarge.

Tests both LNC=2 (4 workers) and LNC=1 (8 workers) configurations.
Run this script on the Neuron instance after deploying the contrib code.

Usage:
    # LNC=2 (default): 4 workers
    python benchmark_dp.py --num-workers 4 --model-path /mnt/models/bioreason-pro-rl

    # LNC=1: 8 workers (must set NEURON_LOGICAL_NC_CONFIG=1 first)
    NEURON_LOGICAL_NC_CONFIG=1 python benchmark_dp.py --num-workers 8 \
        --model-path /mnt/models/bioreason-pro-rl
"""

import argparse
import json
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("benchmark_dp")

# Add src/ to path: test/integration/ -> test/ -> BioReason-Pro/ -> src/
_src_dir = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
)
sys.path.insert(0, _src_dir)

# 10 test proteins (same as all previous benchmarks)
TEST_PROTEINS = [
    {
        "protein_id": "P63089",
        "sequence": (
            "MDPNKLTYQPLTLKHWQEKHPEELSRALGFKELLQEFNAIDYITEVLQEN"
            "TFSKDLETIFKLNEMFSQKSPEELKKQFSKFNPQTGRPNINQKETDSYKVT"
            "LKNMFLNQDISEDYRSFAKTILKAKDLEKEQERYREILNMPNLNEEQRNDL"
            "HMEEVDAHLEEVFNKLHA"
        ),
        "organism": "Mus musculus (Mouse)",
        "interpro": "- IPR000762: Midkine (Mdk), neurite growth-promoting factor 2 [1-168]",
        "gogpt": (
            "GO:0008083 (growth factor activity); GO:0005576 (extracellular region); "
            "GO:0007399 (nervous system development)"
        ),
    },
    {
        "protein_id": "P51864",
        "sequence": (
            "MHQGAPPGVHQLCHPFGSPAAAIAQKLSETPDALGNRQAFRMSLYHEVQAM"
            "LRVFGNLAAGSDVTTRRLHKFHGLEKFPVTKEAIENAFAISKNGKKSVVLM"
            "SHLGRPDGVPMSDSFREMTRDLERGAVNVTFSMSRSHLRKVFPGLHIYTLK"
            "LPKNMEAFTAHQNLITAGLDNGTFVTTTLENGSCFKIHN"
        ),
        "organism": "Homo sapiens (Human)",
        "interpro": "- IPR002300: Aminoacyl-tRNA synthetase, class Ia [10-188]",
        "gogpt": (
            "GO:0004812 (aminoacyl-tRNA ligase activity); GO:0005737 (cytoplasm); "
            "GO:0006412 (translation)"
        ),
    },
    {
        "protein_id": "Q06205",
        "sequence": (
            "MSKMSHFLIYNALDQFIAGDVTPRHTGMIKVYAAELGITLAMQYLIALMSDEGQLATIMV"
            "KPYDKHLALYHEQFVSMNELDDTFPLSKKAKDFSAEVLADKGIEFSFINATMSKSHMFA"
            "MSIAGDKTKGQFFITTKETSAGSLLSMSQHFSAMLKLGLDPNDVDMAPEEADKSKAFHE"
            "FMQSDSSSIDMFENQMMSIHNFTKESQIEEMNSLYASFSEFIHEFNDANNFRSAILNVVD"
            "IATLIHNATDKRSGELFLADRLISSNCPHLTTGQALCMALDALSLISKLNMVSDDMEQLG"
            "TSVSEINFDEPDDGIIFGTIMEGVEALTAATEKKEEADNKPKAKEAQEAANKKGRVDALD"
            "EGDEQIESMFYNQYSDAESKEKIASQRPQAE"
        ),
        "organism": "Saccharomyces cerevisiae (Baker's yeast)",
        "interpro": (
            "- IPR001404: Heat shock protein Hsp90 family (unknown) [1-392]\n"
            "- IPR020575: Heat shock protein 90, conserved site (unknown) [120-150]"
        ),
        "gogpt": (
            "GO:0005524 (ATP binding); GO:0051082 (unfolded protein binding); "
            "GO:0006457 (protein folding); GO:0005737 (cytoplasm)"
        ),
    },
    {
        "protein_id": "A6NI15",
        "sequence": (
            "MKKFRHKKNPVPKKLKNKYNKDDGKMEQYEFLNDKVDLFSKEFYEKSIRFLNKDLELQ"
            "SKCGFPFSYSRHSELHTRIHSGETANFHQFKNIGQEKNAWRFAKNNNRKKNESILKFHD"
            "FKEKFMQTYTAKEGRLNHVEDVTILTLSEEGKFMLESSNVEKHWWDTFHKYKKNKQEWK"
            "AKTSNNKKKSKHYSD"
        ),
        "organism": "Homo sapiens (Human)",
        "interpro": "- IPR012677: Nucleotide-binding, alpha-beta plait [30-193]",
        "gogpt": (
            "GO:0003723 (RNA binding); GO:0005634 (nucleus); "
            "GO:0006397 (mRNA processing)"
        ),
    },
    {
        "protein_id": "Q66LM6",
        "sequence": (
            "MDENVGLPCEDSSALGPAQADLGATFKKMVTRRPKPSAHRKQLSKVLLDMR"
            "KKEEDLKDELGEALQKITHDLKDETDTKLPYEAIQDFLKEHKGLKDDESM"
            "YIFQEVLEKRDDYMTQLQEELAHFEAEIRTHTENKELRQLYKQVKQKIELD"
            "KRCLNPPDTKYNRFYYVNDDYEEFNPHRNSHKQTGTLRKTFNILLERGKN"
            "GFISVEAAFADQYLVEIYQKMAQEDHRKGYLQVLNELRGFPEDPSPFGAFS"
            "KVLANQVLAQFEPDQAQEFTHVMEKTIHCFQKTPENREQEYNAIRQNLDKN"
            "RDGTIIDLSTTELDTTFNTPLPSQVQFP"
        ),
        "organism": "Mus musculus (Mouse)",
        "interpro": "- IPR001680: WD40 repeat [100-333]",
        "gogpt": (
            "GO:0005515 (protein binding); GO:0005737 (cytoplasm); "
            "GO:0016567 (protein ubiquitination)"
        ),
    },
    {
        "protein_id": "P0C2H9",
        "sequence": (
            "MSKMSHFLIYNALDQFIAGDVTPRHTGMIKVYAAELGITLAMQYLIALMSDEG"
            "QLATIMVKPYDKHLALYHEQFVSMNELDDTFPLSKKAKDFSAEVLADKGIE"
            "FSFINAT"
        ),
        "organism": "Saccharomyces cerevisiae (Baker's yeast)",
        "interpro": (
            "- IPR001404: Heat shock protein Hsp90 family (unknown) [1-113]\n"
            "- IPR020575: Heat shock protein 90, conserved site (unknown) [21-33]"
        ),
        "gogpt": (
            "GO:0005524 (ATP binding); GO:0051082 (unfolded protein binding); "
            "GO:0006457 (protein folding); GO:0005737 (cytoplasm)"
        ),
    },
    {
        "protein_id": "Q9M8K6",
        "sequence": (
            "MEILSQNLNFSQFSRENISDYTQYYYSEQPSEVFTKFLRQEIATLHKQNNE"
            "LGSVLGIGTIVYSTMALILRNLMQEKANKNEISKDKIKHFHKSLNELQSFN"
            "IHLRLSEELRCLVNEHLKEMADLSRHYTSQPEYTDMPVVKLKRLISALNYD"
            "LPHISQPDFSQENSNIMISYFAGPKDQMAKEGQFIRFHAVFCNSGADFSKY"
        ),
        "organism": "Arabidopsis thaliana (Mouse-ear cress)",
        "interpro": "- IPR001128: Cytochrome P450 [1-202]",
        "gogpt": (
            "GO:0004497 (monooxygenase activity); GO:0005506 (iron ion binding); "
            "GO:0016705 (oxidoreductase activity); GO:0005789 (endoplasmic reticulum membrane)"
        ),
    },
    {
        "protein_id": "Q8W488",
        "sequence": (
            "MSLLPSLFPRFFKNSSKKPLFYLFFICLSFSMSAATDVHEFNMSQMEQFDN"
            "MSENITSQPFLNRNYISQDPYTFLTYYAANRPEALKKYASENPDDSNEIFR"
            "AKNFLFNLDQFESPNFKKNWIQFQDQFQNLSQINSENSQFQFNYSAFNYS"
            "MYAFKDEQLSQFIKYEDSNQWIFFQQDKKYEFIKKFSNEKFDALKNKYPNL"
            "DTFQRKNLNNLTSEFRRFCMKDDLLLEQIDLIPHIPQNQSLYASFSFSPS"
            "SSTFTAQPNFNYNFYNYNFSNFNQHNESSPFFDANNPLFEQSSTFFEQPK"
            "LQNFSFQQPQFGTCFKMAQDVGSVFGSLLMLGAAELRSRVFSNKFHYQFM"
            "AQYAGVMMFGAALLMRLPLVFAEPFGNYLMMDQTQFALLGGSQVMFMQHQ"
            "QQNQSQETLFNQQLQFIAENANLVIFITSLFLFYLFIIQEEKIFRFNSQFN"
            "AVLRQMASKLSIFVAITLMIAAIAGFYLSTHQYLKNPEKESQA"
        ),
        "organism": "Arabidopsis thaliana (Mouse-ear cress)",
        "interpro": "- IPR002528: MATE efflux family protein [50-494]",
        "gogpt": (
            "GO:0015238 (drug transmembrane transporter activity); "
            "GO:0016021 (integral component of membrane); "
            "GO:0055085 (transmembrane transport)"
        ),
    },
    {
        "protein_id": "P0A9K3",
        "sequence": (
            "MNIFEIDHSGGELVTTFEGNEIRVRQIDGFEVTKMMRLSEQRREALESALGA"
            "VSMYSVYQEKVNARLKALYELAEGDIEGIRIPYRIFNNGSDVHQALINGEE"
            "AVKFYDMCPYYSFRECKLNLPETTLVHSISGNFNLASDAFDLGGSFTPMT"
            "VSQHQQLLHKIMSAKYQDDNVSSVTFEDTKEMFARSQAVDQHVLTTCDYR"
            "TYMCFDDLVPGLRQELLHCPFISSQIIFTVHENGTNFSTQYEQMFADQTR"
            "LGVRDRSQAKNQFSFAVNFCSAFAQAECAGYFEKYNIQAGFQAIYPQSFSI"
            "MNTMYNHKPQAEHIMQFFETFNQEDISAFEKNTMNHISP"
        ),
        "organism": "Escherichia coli (strain K12)",
        "interpro": "- IPR005999: Glycerol kinase [1-346]",
        "gogpt": (
            "GO:0004370 (glycerol kinase activity); GO:0005524 (ATP binding); "
            "GO:0019563 (glycerol catabolic process); GO:0005737 (cytoplasm)"
        ),
    },
    {
        "protein_id": "tmem106a",
        "sequence": (
            "MEPRWWLAALGSLLLPAPLLLTSGEEDTPRCVHGMKCQDQELQELVLGPGE"
            "KDDRLFLHKDGLMLEINPDSGAHLQMEELVDRYADNLSLQVKEDPDDYSS"
            "KHLHLRQEKLQAQHIFQETFNQLEIATTLDRESHLVLHRCEVLNRALEFKE"
            "LLQNPYLGAPLQGSTNSPQFMQYNLSSQNGHMDNMTYTAMDPTPQRKNPK"
            "ATANPQSSSYPYLGYGQEPQHAQPTLEQANFISAQRQALQLGLQASSQEKL"
            "QALEGLAK"
        ),
        "organism": "Homo sapiens (Human)",
        "interpro": "- IPR006574: Transmembrane protein 106 [1-262]",
        "gogpt": (
            "GO:0016021 (integral component of membrane); "
            "GO:0005764 (lysosome); GO:0005886 (plasma membrane)"
        ),
    },
]


def main():
    parser = argparse.ArgumentParser(description="Benchmark DP BioReason-Pro")
    parser.add_argument(
        "--model-path", required=True, help="Path to bioreason-pro-rl checkpoint"
    )
    parser.add_argument(
        "--num-workers", type=int, required=True, help="Number of DP workers (cores)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Per-worker batch size"
    )
    parser.add_argument(
        "--compiled-model-path", default=None, help="Path for compiled artifacts"
    )
    parser.add_argument("--max-context-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--esm3-model", default="esm3_sm_open_v1")
    parser.add_argument("--start-core", type=int, default=0, help="First core ID")
    parser.add_argument("--output-json", default=None, help="Save results to JSON file")
    args = parser.parse_args()

    from dp_launcher import DataParallelRunner

    compiled_path = (
        args.compiled_model_path
        or f"/mnt/compiled/dp{args.num_workers}_bs{args.batch_size}"
    )

    log.info(f"=== BioReason-Pro DP Benchmark ===")
    log.info(f"Workers: {args.num_workers}, BS/worker: {args.batch_size}")
    log.info(f"Model: {args.model_path}")
    log.info(f"Compiled: {compiled_path}")
    log.info(f"LNC config: {os.environ.get('NEURON_LOGICAL_NC_CONFIG', '2 (default)')}")
    log.info(f"Proteins: {len(TEST_PROTEINS)}")

    runner = DataParallelRunner(
        model_path=args.model_path,
        num_workers=args.num_workers,
        esm3_model=args.esm3_model,
        max_context_length=args.max_context_length,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        tp_degree=1,
        compiled_model_path=compiled_path,
        start_core=args.start_core,
    )

    log.info("Starting benchmark...")
    bench = runner.benchmark(TEST_PROTEINS)

    log.info(f"\n{'=' * 60}")
    log.info(f"RESULTS: {args.num_workers} workers, BS={args.batch_size}/worker")
    log.info(f"{'=' * 60}")
    log.info(f"Wall time:           {bench['wall_time_s']:.1f}s")
    log.info(f"Total tokens:        {bench['total_tokens']}")
    log.info(f"Aggregate tok/s:     {bench['aggregate_tok_s']:.1f}")
    log.info(f"Mean worker tok/s:   {bench['mean_worker_tok_s']:.1f}")
    log.info(f"Errors:              {bench['num_errors']}")
    log.info(f"")
    log.info(f"Per-worker tok/s:")
    for wid, tps in sorted(bench["per_worker_tok_s"].items()):
        log.info(f"  Worker {wid}: {tps:.1f} tok/s")

    # Per-protein summary
    log.info(f"\nPer-protein results:")
    for r in bench["results"]:
        if "error" in r:
            log.info(f"  [{r['protein_idx']}] ERROR: {r['error']}")
        else:
            pid = TEST_PROTEINS[r["protein_idx"]]["protein_id"]
            log.info(
                f"  [{r['protein_idx']}] {pid}: "
                f"{r['num_tokens']} tok, {r['gen_time_s']:.1f}s gen, "
                f"{r['tok_per_s']:.1f} tok/s (worker {r['worker_id']}, core {r['core_id']})"
            )

    if args.output_json:
        # Serialize results (remove full text for compact output)
        output = {
            "config": {
                "num_workers": args.num_workers,
                "batch_size": args.batch_size,
                "max_context_length": args.max_context_length,
                "max_new_tokens": args.max_new_tokens,
                "lnc_config": os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"),
            },
            "wall_time_s": bench["wall_time_s"],
            "total_tokens": bench["total_tokens"],
            "aggregate_tok_s": bench["aggregate_tok_s"],
            "mean_worker_tok_s": bench["mean_worker_tok_s"],
            "per_worker_tok_s": bench["per_worker_tok_s"],
            "num_errors": bench["num_errors"],
            "per_protein": [
                {
                    "protein_id": TEST_PROTEINS[r["protein_idx"]]["protein_id"],
                    "num_tokens": r.get("num_tokens", 0),
                    "gen_time_s": r.get("gen_time_s", 0),
                    "tok_per_s": r.get("tok_per_s", 0),
                    "worker_id": r.get("worker_id", -1),
                    "error": r.get("error"),
                }
                for r in bench["results"]
            ],
        }
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        log.info(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
