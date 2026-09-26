import React, { useEffect } from 'react';
import { Button  } from 'antd';
import Paragraph from '../../components/Paragraph';
import { FileTextOutlined } from '@ant-design/icons';
import { Link } from 'react-router-dom';
import { SmallGithubIcon, HFIcon } from '../../components/Icons';
import Video from '../../components/Video';
import {
    Container,
    ContentOuter,
    ContentInner,
    Banner,
    BadgeContainer,
    Children,
    Anchor,
    PaperTitle,
    PaperShortDescription,
    PaperAuthors,
    PaperAuthorOrganizations,
    Title
} from '../Project/Parent.style.js';
import Image from '../../components/Image';


import shrimpIcon from '../../images/tempura/shrimp.png';
import TEMUPRA_Teaser from '../../images/tempura/teaser.png';
import TEMUPRA_Training_Pipeline from '../../images/tempura/training_pipeline.png';
import TEMUPRA_Data_Pipeline from '../../images/tempura/data_pipeline.png';
import TEMUPRA_Data_Example from '../../images/tempura/sft_data_example.png';

const TEMPURA = () => {
    useEffect(() => {
        const originalTitle = document.title;
        document.title = 'TEMPURA';
        
        return () => {
            document.title = originalTitle;
        };
    }, []);

    // Author order and affiliations follow the COLM 2026 camera-ready (arXiv:2505.01583).
    const authors = [
        { name: "Jen-Hao Cheng", affiliation: 1, link: "https://jen-haocheng.com/" },
        { name: "Yi-Hao Peng", affiliation: 2, link: "https://www.yihaopeng.tw/" },
        { name: "Huapeng Zhou", affiliation: 1, link: "https://huapengzhou.com/" },
        { name: "Vivian Wang", affiliation: 1, link: "https://www.linkedin.com/in/vivian-wang-bb14a4225/" },
        { name: "Huayu Wang", affiliation: 1, link: "https://huayuww.github.io/" },
        { name: "Hsiang-Wei Huang", affiliation: 1, link: "https://hsiangwei0903.github.io/" },
        { name: "Wenhao Chai", affiliation: 3, link: "https://wenhaochai.com/" },
        { name: "Hou-I Liu", affiliation: 4, link: "https://www.linkedin.com/in/hoiliu0801/" },
        { name: "Kuang-Ming Chen", affiliation: 1, link: "https://gorden0413.github.io/" },
        { name: "Cheng-Yen Yang", affiliation: 1, link: "https://yangchris11.github.io/" },
        { name: "Yi-Ling Chen", affiliation: 5, link: "https://www.linkedin.com/in/yiling-chen-tw" },
        { name: "Vibhav Vineet", affiliation: 5, link: "https://vibhav-vineet.github.io/" },
        { name: "Qin Cai", affiliation: 6, link: "https://www.linkedin.com/in/qin-cai-4329a195" },
        { name: "Jenq-Neng Hwang", affiliation: 1, link: "https://people.ece.uw.edu/hwang/" }
    ];

    const affiliations = [
        { id: 1, name: "University of Washington" },
        { id: 2, name: "Carnegie Mellon University" },
        { id: 3, name: "Princeton University" },
        { id: 4, name: "National Yang Ming Chiao Tung University" },
        { id: 5, name: "Microsoft" },
        { id: 6, name: "Independent Researcher" }
    ];

    return (
        <Container>
            <ContentOuter>
                <ContentInner>
                    {/* <Typography.Title level={1} style={{ textAlign: "center", fontSize: 48 }}>
                    <div style={{ display: "flex", justifyContent: "center", alignItems: "center" }}>
                        <img src={shrimpIcon} alt="shrimp" style={{ width: 48, height: 48 }} />
                        TEMPURA: Temporal Event Masked Prediction
                        and Understanding for Reasoning in Action
                    </div>
                    </Typography.Title>  */}

                    <PaperTitle >
                            <img src={shrimpIcon} alt="shrimp" style={{ width: 36, height: 36 }} />
                            TEMPURA: Temporal Event Masked Prediction
                            and Understanding for Reasoning in Action
                    </PaperTitle> 
                    <div style={{ textAlign: "center", fontSize: "18px", fontWeight: "500", color: "#8c8c8c", marginBottom: "8px" }}>
                        Conference on Language Modeling (COLM) 2026
                    </div>
                    <PaperShortDescription>
                    TEMPURA enables video-language models to reason about causal event relationships and generate fine-grained, timestamped descriptions of untrimmed videos.
                    </PaperShortDescription>
                    <PaperAuthors>
                        {
                            authors.map((author) => (
                                <div key={author.name}>
                                    <a href={author.link}>{author.name}</a>
                                    <sup>{author.affiliation}</sup>
                                </div>
                            ))
                        }
                    </PaperAuthors>
                    <PaperAuthorOrganizations>
                        {
                            affiliations.map((affiliation) => (
                                <div key={affiliation.id}>
                                    <sup>{affiliation.id}</sup>
                                    <span>{affiliation.name}</span>
                                </div>
                            ))
                        }
                    </PaperAuthorOrganizations>

                    <div style={{ display: "flex", justifyContent: "center", alignItems: "center", flexWrap: "row wrap", marginTop: "40px" }}>
                        <Button size="medium" style={{ display: 'flex', alignItems: 'center', marginRight: "8px"}} href="https://arxiv.org/abs/2505.01583">
                            <FileTextOutlined style={{ marginRight: "4px", display: 'flex', alignItems: 'center', marginTop: "8px" }}/>
                            <span style={{ marginTop: "8px" }}>Paper</span>
                        </Button>

                        <Button size="medium" style={{ display: 'flex', alignItems: 'center', marginRight: "8px"}} href="https://github.com/Andy-Cheng/TEMPURA">
                            <SmallGithubIcon style={{ marginRight: "4px", display: 'flex', alignItems: 'center', marginTop: "8px" }}/>
                            <span style={{ marginTop: "8px" }}>Code</span>
                        </Button>

                        <Button size="medium" style={{ display: 'flex', alignItems: 'center', marginRight: "8px"}} href="https://huggingface.co/datasets/andaba/TEMPURA-VER">
                            <HFIcon style={{ marginRight: "4px", display: 'flex', alignItems: 'center', marginTop: "8px" }}/>
                            <span style={{ marginTop: "8px" }}>Data</span>
                        </Button>

                        <Button size="medium" style={{ display: 'flex', alignItems: 'center'}} href="https://huggingface.co/collections/andaba/tempura-681c325777c23f72666a0995">
                            <HFIcon style={{ marginRight: "4px", display: 'flex', alignItems: 'center', marginTop: "8px" }}/>
                            <span style={{ marginTop: "8px" }}>Models</span>
                        </Button>
                    </div>

                    
                    {/* <Avatar style={{ marginLeft: "50%", transform: "translateX(-50%)" }} size={150} src={AndyImg} /> */}

                    {/* <Typography.Title level={5} style={{ color: "gray", marginBottom: "10px", marginTop: "40px", textAlign: "center" }}>
                        
                    </Typography.Title> */}

                    <Title style={{ marginTop: "32px" }}>
                        Paper Abstract
                    </Title>
                    <Paragraph>
                    Understanding causal event relationships and achieving fine-grained temporal grounding in videos remain challenging for vision-language models (VLMs). We propose TEMPURA (Temporal Event Masked Prediction and Understanding for Reasoning in Action), a two-stage training framework that enhances the video temporal understanding of VLMs. Inspired by infilling techniques in language modeling, TEMPURA first performs masked event prediction, learning to reconstruct missing events and generate step-by-step causal explanations from dense event annotations. It then learns video segmentation and dense captioning, decomposing videos into non-overlapping events with detailed, timestamp-aligned descriptions. We train TEMPURA on VER, our large-scale dataset of 500K videos annotated with temporally aligned event descriptions and structured reasoning steps. Experiments on video temporal grounding and highlight detection benchmarks show that TEMPURA substantially improves strong base VLMs across model families and scales, confirming that combining event-level reasoning with fine-grained temporal segmentation is an effective recipe for video temporal understanding.
                    </Paragraph>
                    
                    <Image src={TEMUPRA_Teaser} />

                    <Paragraph>
                    TEMPURA uses a two-stage process for video understanding. The model
first infers event structures and causal relationships by filling in missing details and reasoning about event
sequences (e.g., recognizing that shrimp must be battered before frying). Second, it is learned to partition video
into non-overlapping events and describe them in details.
                    </Paragraph>

                    <Title style={{ marginTop: "32px" }}>
                        TEMPURA: Two-Stage Training Pipeline
                    </Title>
                    <Paragraph>
                    TEMPURA’s two-stage training: (a) Masked Event Prediction infers missing events with causal reasoning; (b) Temporal Segmentation divides videos into timestamped, non-overlapping events with detailed captions for structured understanding.
                    </Paragraph>
                    <Image src={TEMUPRA_Training_Pipeline} />

                    <Title style={{ marginTop: "32px" }}>
                        VER Dataset: Powering Fine-Grained Temporal Understanding
                    </Title>
                    <Paragraph>
                    We constructed the Video Event Reasoning (VER) dataset—500K untrimmed videos totaling 18K hours—each densely annotated with timestamp-aligned, non-overlapping events and detailed captions. VER addresses limitations in existing datasets by providing full video coverage and fine-grained event segmentation, enabling TEMPURA to (1) segment videos comprehensively, (2) describe each event in detail, and (3) reason about missing events using contextual cues.

                    </Paragraph>
                    <Paragraph>
                    The pipeline begins by filtering and categorizing a large video pool. GPT-4o then
generates event captions with start/end times, followed by a temporal coherence check that discards invalid
events. For valid events, a subset is masked to form a fill-in-the-blank task, and GPT-4o infers the missing
segments—ultimately creating a dataset for video temporal understanding.
                    </Paragraph>

                    <Image src={TEMUPRA_Data_Pipeline} />

                    <Paragraph>
                        Here is an example from the VER dataset.
                    </Paragraph>
                    <Image src={TEMUPRA_Data_Example} />

                    <Title style={{ marginTop: "32px" }}>
                        TEMPURA Model
                    </Title>
                    <Paragraph>
                    We train TEMPURA on the VER dataset on top of Qwen2.5-VL (3B, 7B) and InternVL3 (2B, 8B). Compared to the base models, TEMPURA provides more precise timestamps and generates finer-grained event descriptions. Checkpoints, inference code and the benchmark-evaluation pipeline are released on <a href="https://huggingface.co/collections/andaba/tempura-681c325777c23f72666a0995">Hugging Face</a> and <a href="https://github.com/Andy-Cheng/TEMPURA">GitHub</a>. In the following video examples, the closed captions are generated by the models.
                    </Paragraph>
                    <Title style={{ marginTop: "8px", fontSize: "24px" }}>
                        Examples
                    </Title>
                    <Paragraph style={{ marginTop: "16px", fontWeight: "300", fontFamily: "monospace", color: "#ff4d4f", marginBottom: "16px" }}>
                        Please enable Closed Captions (CC) to view their time-aligned event descriptions.
                    </Paragraph>
                    <Paragraph style={{ marginTop: "16px", fontWeight: "300", fontFamily: "monospace", color: "#d48806" }}>
                        Luke Skywalker fighting scene in The Mandalorian.
                    </Paragraph>
                    <div style={{ fontSize: "16px", fontWeight: "bold", marginBottom: "8px", color: "#434343", marginTop: "0px" }}>
                        TEMPURA-Qwen2.5-VL-3B
                    </div>
                    <iframe width="100%" height="540" src="https://www.youtube-nocookie.com/embed/S7w3TKQf06M?si=t64DidONDROq9YId" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
                    <Paragraph>
                        TEMPURA generates more accurate timestamps and more detailed event descriptions.
                    </Paragraph>
                    <div style={{ fontSize: "16px", fontWeight: "bold", marginBottom: "8px", color: "#434343", marginTop: "32px" }}>
                        Qwen2.5-VL-3B
                    </div>
                    <iframe width="100%" height="540" src="https://www.youtube-nocookie.com/embed/mBIDSU4dlvU?si=M2Uf0NIHU-40yGWM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
                    <Paragraph>
                        In contrast, the baseline model, Qwen2.5-VL-3B, struggles to generate accurate timestamps and detailed event descriptions.
                    </Paragraph>

                    <Paragraph style={{ marginTop: "32px", fontWeight: "300", fontFamily: "monospace", color: "#d48806" }}>
                        How to make keto hot dogs?
                    </Paragraph>
                    <div style={{ fontSize: "16px", fontWeight: "bold", marginBottom: "8px", color: "#434343" }}>
                        TEMPURA-Qwen2.5-VL-3B
                    </div>
                    <iframe width="100%" height="540" src="https://www.youtube-nocookie.com/embed/-afEHRkV7J8?si=wxx69XRHpErSmF18" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
                    <div style={{ fontSize: "16px", fontWeight: "bold", marginBottom: "8px", color: "#434343", marginTop: "32px" }}>
                        Qwen2.5-VL-3B
                    </div>
                    <iframe width="100%" height="540" src="https://www.youtube-nocookie.com/embed/hp5oXz7ZDm0?si=6gQ91a2BLiKdHJ1s" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

                    <Title style={{ marginTop: "64px", fontSize: "24px", marginBottom: "16px" }}>
                        Demo APP
                    </Title>
                    <iframe width="100%" height="540" src="https://www.youtube-nocookie.com/embed/p979hTHKJoE?si=8k8bUyb67_JAwhk9" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
                    <Paragraph>
                        Please check out the paper for quantitative results on Charades-STA temporal grounding and QVHighlights highlight detection, and the GitHub repository for the end-to-end inference and evaluation pipeline.
                    </Paragraph>      

                    <div style={{ color: "black", marginBottom: "12px", marginTop: "12px", textAlign: "center", fontSize: "24px", fontWeight: "400" }}>
                        BibTeX
                    </div>
                    <div style={{ 
                        color: "black", 
                        marginBottom: "12px", 
                        marginTop: "12px", 
                        fontSize: "12px", 
                        fontWeight: "300",
                        fontFamily: "monospace",
                        whiteSpace: "pre-wrap",
                        backgroundColor: "#f0f0f0",
                        padding: "16px",
                        borderRadius: "4px",
                        padding: "32px 32px",
                    }}>
                        {`@inproceedings{
cheng2026tempura,
title={{TEMPURA}: Temporal Event Masked Prediction and Understanding for Reasoning in Action},
author={Cheng, Jen-Hao and Peng, Yi-Hao and Zhou, Huapeng and Wang, Vivian and Wang, Huayu and Huang, Hsiang-Wei and Chai, Wenhao and Liu, Hou-I and Chen, Kuang-Ming and Yang, Cheng-Yen and Chen, Yi-Ling and Vineet, Vibhav and Cai, Qin and Hwang, Jenq-Neng},
booktitle={Third Conference on Language Modeling},
year={2026}
}`}
                    </div>
                </ContentInner>
            </ContentOuter>
        </Container>
    );
};

export default TEMPURA;
