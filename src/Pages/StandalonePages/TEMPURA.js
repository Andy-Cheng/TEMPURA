import React, { useEffect } from 'react';
import { Button  } from 'antd';
import Paragraph from '../../components/Paragraph';
import { FileTextOutlined } from '@ant-design/icons';
import { Link } from 'react-router-dom';
import { SmallGithubIcon, HFIcon } from '../../components/Icons';
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

    const authors = [
        {
            name: "Jen-Hao Cheng",
            link: "https://jen-haocheng.com/",
            affiliation: 1
        },
        {
            name: "Vivian Wang",
            affiliation: 1,
            link: "https://www.linkedin.com/in/vivian-wang-bb14a4225/"

        },
        {
            name: "Huayu Wang",
            affiliation: 1,
            link: "https://www.linkedin.com/in/huayu-wang-5a68981b6/"
        },
        {
            name: "Huapeng Zhou",
            affiliation: 1,
            link: "https://huapengzhou.github.io/"
        },
        {
            name: "Yi-Hao Peng",
            affiliation: 2,
            link: "https://www.yihaopeng.tw/"
        },
        {
            name: "Hou-I Liu",
            affiliation: 3,
            link: "https://scholar.google.com.tw/citations?user=7L8K_o0AAAAJ&hl=zh-TW"
        },
        {
            name: "Hsiang-Wei Huang",
            affiliation: 1,
            link: "https://hsiangwei0903.github.io/"
        },
        {
            name: "Kuang-Ming Chen",
            affiliation: 1,
            link: "https://gorden0413.github.io/"
        },
        {
            name: "Cheng-Yen Yang",
            affiliation: 1,
            link: "https://yangchris11.github.io/"
        },
        {
            name: "Wenhao Chai",
            affiliation: 1,
            link: "https://wenhaochai.com/"
        },
        {
            name: "Yi-Ling Chen",
            affiliation: 4,
            link: "https://scholar.google.com/citations?user=jI5oDhYAAAAJ"
        },
        {
            name: "Vibhav Vineet",
            affiliation: 4,
            link: "https://vibhav-vineet.github.io/"
        },
        {
            name: "Qin Cai",
            affiliation: 5,
            link: "https://scholar.google.com/citations?user=M_fwZg0AAAAJ"
        },
        {
            name: "Jenq-Neng Hwang",
            affiliation: 1,
            link: "https://people.ece.uw.edu/hwang/"
        }
    ];

    const affiliations = [
        {
            id: 1,
            name: "University of Washington"
        },
        {
            id: 2,
            name: "Carnegie Mellon University"
        },
        {
            id: 3,
            name: "National Yang Ming Chiao Tung University"
        },
        {
            id: 4,
            name: "Microsoft"
        },
        {
            id: 5,
            name: "Independent Researcher"
        }
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
                        <Button size="medium" style={{ display: 'flex', alignItems: 'center', marginRight: "8px"}} href="">
                            <FileTextOutlined style={{ marginRight: "4px", display: 'flex', alignItems: 'center', marginTop: "8px" }}/>
                            <span style={{ marginTop: "8px" }}>Preprint</span>
                        </Button>

                        <Button size="medium" style={{ display: 'flex', alignItems: 'center', marginRight: "8px"}} href="https://github.com/Andy-Cheng/TEMPURA">
                            <SmallGithubIcon style={{ marginRight: "4px", display: 'flex', alignItems: 'center', marginTop: "8px" }}/>
                            <span style={{ marginTop: "8px" }}>Code</span>
                        </Button>

                        <Button size="medium" style={{ display: 'flex', alignItems: 'center'}} href="https://huggingface.co/datasets/andaba/TEMPURA-VER">
                            <HFIcon style={{ marginRight: "4px", display: 'flex', alignItems: 'center', marginTop: "8px" }}/>
                            <span style={{ marginTop: "8px" }}>Data</span>
                        </Button>
                    </div>

                    
                    {/* <Avatar style={{ marginLeft: "50%", transform: "translateX(-50%)" }} size={150} src={AndyImg} /> */}

                    {/* <Typography.Title level={5} style={{ color: "gray", marginBottom: "10px", marginTop: "40px", textAlign: "center" }}>
                        
                    </Typography.Title> */}

                    <Title style={{ marginTop: "32px" }}>
                        Paper Abstract
                    </Title>
                    <Paragraph>
                    Understanding causal event relationships and achieving fine-grained temporal grounding in videos remain challenging for vision-language models. Existing methods either compress video tokens to reduce temporal resolution, or treat videos as unsegmented streams, which obscures fine-grained event boundaries and limits the modeling of causal dependencies. We propose TEMPURA (Temporal Event Masked Prediction and Understanding for Reasoning in Action), a two-stage training framework that enhances video temporal understanding. TEMPURA first applies masked event prediction reasoning to reconstruct missing events and generate step-by-step causal explanations from dense event annotations, drawing inspiration from effective infilling techniques. TEMPURA then learns to perform video segmentation and dense captioning to decompose videos into non-overlapping events with detailed, timestamp-aligned descriptions. We train TEMPURA on VER, a large-scale dataset curated by us that comprises 1M training instances and 500K videos with temporally aligned event descriptions and structured reasoning steps. Experiments on temporal grounding and highlight detection benchmarks demonstrate that TEMPURA outperforms strong baseline models, confirming that integrating causal reasoning with fine-grained temporal segmentation leads to improved video understanding.
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
                        We finetuned Qwen2.5-VL on VER to train TEMPURA. TEMPURA delivers precise timestamps, fine-grained segmentation, and rich captions—outperforming other models, as shown by the following examples.
                    </Paragraph>

                    <Paragraph>
                        Please check out the paper for more quantitative results.
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
                        borderRadius: "4px"
                    }}>
                        {`@article{tempura,
    title={TEMPURA: Temporal Event Masked Prediction and Understanding for Reasoning in Action},
    author={Jen-Hao Cheng and Vivian Wang and Huayu Wang and Huapeng Zhou and Yi-Hao Peng and Hou-I Liu and Hsiang-Wei Huang and Kuang-Ming Chen and Cheng-Yen Yang and Wenhao Chai and Yi-Ling Chen and Vibhav Vineet and Qin Cai and Jenq-Neng Hwang},
    year={2025}
}`}
                    </div>
                </ContentInner>
            </ContentOuter>
        </Container>
    );
};

export default TEMPURA;
