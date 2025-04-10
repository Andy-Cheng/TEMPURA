import React from 'react';
import Item from './Item';
import * as PROJECTS from '../Project/projects.const';

import CenterRadarNetCover from '../../images/cenrad_banner.jpg';
import GameIllusionizationCover from '../../images/game-illusionization_cover.jpg';
import hapticGoRoundCover from '../../images/hapticGoRoundImg1.jpg';
import impossibleStaircaseCover from '../../images/impossible-staircase_cover.jpg';
import CRUW3DCover from '../../images/CRUW3D_cover.jpg';
import CRAFTBoosterCover from '../../images/CrossCheck_Cover.jpg';
import RTPCover from '../../images/rtpose_cover.jpg';
import MonoTAKD from '../../images/MonoTAKD_cover.jpg';


const WORKS = [
  {
    title: 'MonoTAKD',
    description: "Published on CVPR '25",
    mediaSrc: MonoTAKD,
    path: PROJECTS.MONOTAKD
  },
  {
    title: 'RT-Pose',
    description: "Published on ECCV '24",
    mediaSrc: RTPCover,
    path: PROJECTS.RTPOSE
  },
  {
    title: 'CRAFTBooster',
    description: "Published on IEEE IV '24",
    mediaSrc: CRAFTBoosterCover,
    path: PROJECTS.CRAFTBOOSTER
  },
  {
    title: 'Vision meets mmWave Radar',
    description: "Published on IEEE IV '24",
    mediaSrc: CRUW3DCover,
    path: PROJECTS.CRUW3D
  },
  {
    title: 'CenterRadarNet',
    description: "Published on IEEE ICIP '24",
    mediaSrc: CenterRadarNetCover,
    path: PROJECTS.CENTER_RADAR_NET
  },
  {
    title: 'Game Illusionization',
    description: "Published on ACM UIST '21",
    mediaSrc: GameIllusionizationCover,
    path: PROJECTS.GAME_ILLUSIONIZATION
  },
  {
    title: 'Impossible Staircase',
    description: "Published on IEEE VR '21",
    mediaSrc: impossibleStaircaseCover,
    path: PROJECTS.IMPOSSIBLE_STAIRCASE
  },
  {
    title: 'Haptic-go-round',
    description: "Published on ACM CHI '20, SIGGRAPH '20 Immersive Pavilion",
    mediaSrc: hapticGoRoundCover,
    path: PROJECTS.HAPTIC_GO_ROUND
  }
];

const Works = () => (
  <div style={{ display: 'flex', flexWrap: 'wrap' }}>
    {WORKS.map(({ title, description, mediaSrc, path, index }) => (
      <Item
        title={title}
        description={description}
        mediaSrc={mediaSrc}
        path={path}
        key={index}
      />
    ))}
  </div>
);

export default Works;
