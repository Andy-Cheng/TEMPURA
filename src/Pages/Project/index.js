import React from 'react';
import { withRouter } from 'react-router';
import ComingSoon from '../ComingSoon';
import * as PROJECTS from './projects.const';
import HapticGoRound from './HapticGoRound';
import CenterRadarNet from './CenterRadarNet';
import ExternalLink from '../../components/ExternalLink';
import ImpossibleStaircase from './ImpossibleStaircase';
const Project = ({ location }) => {
  const projectName = location.hash.substring(1);
  switch (projectName) {
    case PROJECTS.HAPTIC_GO_ROUND:
      return <HapticGoRound />;
    case PROJECTS.CENTER_RADAR_NET:
      return <CenterRadarNet />;
    case PROJECTS.GAME_ILLUSIONIZATION:
      return <ExternalLink title="Game Illusionization" link="https://poyaocosmoswang.github.io/works/game_illusionization.html"/>;
    case PROJECTS.IMPOSSIBLE_STAIRCASE:
      return <ImpossibleStaircase />;
    case PROJECTS.CRUW3D:
      return <ExternalLink title="Vision meets mmWave Radar" link="https://arxiv.org/abs/2311.10261"/>;;
    case PROJECTS.CRAFTBOOSTER:
      return <ExternalLink title="CRAFTBooster" link="https://arxiv.org/abs/2407.13937"/>;
    case PROJECTS.RTPOSE:
      return <ExternalLink title="RT-Pose" link="https://github.com/ipl-uw/RT-POSE"/>;
    case PROJECTS.MONOTAKD:
      return <ExternalLink title="MonoTAKD" link="https://arxiv.org/abs/2404.04910"/>;
    default:
      return <ComingSoon />;
  }
};

export default withRouter(Project);
