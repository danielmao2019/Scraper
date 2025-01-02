import re


PUB_NAME_MAPPING = {
    'CoRR': 'CoRR',
    'CVPR': 'Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition',
    'ICCV': 'Proceedings of the IEEE/CVF International Conference on Computer Vision',
    'ECCV': 'ECCV',
    'ACCV': 'ACCV',
    'WACV': 'WACV',
    'ICML': 'ICML',
    'ICLR': 'ICLR',
    'NIPS': 'Advances in Neural Information Processing Systems',
    'NeurIPS': 'Advances in Neural Information Processing Systems',
    'SIGGRAPH Asia': 'SIGGRAPH Asia',
    'SIGGRAPH': 'SIGGRAPH',
    'Robotics: Science and Systems': 'RSS',
    'Journal of Computing in Civil Engineering': 'Journal of Computing in Civil Engineering',
    'Computing in Civil Engineering': 'Computing in Civil Engineering',
    'TMLR': 'TMLR',
    'Remote Sensing': 'Remote Sensing',
    'Engineering Structures': 'Engineering Structures',
}


def parse_pub_name(string: str) -> str:
    for pub_name in PUB_NAME_MAPPING.keys():
        matches = re.findall(
            pattern=pub_name, string=string, flags=re.IGNORECASE,
        )
        if len(matches) == 1 and matches[0].lower() == pub_name.lower():
            return PUB_NAME_MAPPING[pub_name]
    raise RuntimeError(f"Cannot parse pub name from \"{string}\".")
