import React from 'react';
import Parent from './Parent';
import peryBannerSrc from '../../images/peryBannerSrc.jpg';

const Pery = () => (
  <Parent
    bannerSrc={peryBannerSrc}
    title="H. Spectrum Accelerator / Incubator Training Program"
    date="Jan. 2018 - July. 2018"
    location="Taipei, Taiwan"
    category="Extracurricular Activity"
    responsibilities="Tech Lead"
    honors="Selected as one of teams presenting on the final demo day."
  >
    yoyoyo
    <div>This is Pery</div>
  </Parent>
);

export default Pery;
