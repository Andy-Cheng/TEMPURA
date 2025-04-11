# Andy's Personal Website
This is my personal website built with [React](https://reactjs.org/) and [Ant Design](https://ant.design/).
Link to the website: [jen-haocheng.com](https://jen-haocheng.com)

Feel free to use the code for your own website.



## How to run

```bash
npm install
npm start
```



## How to deploy

1. Change your homepage in `package.json` to your github page domain.

2. Run the following command to deploy your website.

```bash
npm run deploy_gh
```

The website will be deployed to `https://<your github username>.github.io`.


3. (Optional) Setup a custom domain by adding a `CNAME` file to the `build` folder.

First, you need to buy a domain name and point it to your github page.
I used [GoDaddy](https://www.godaddy.com/) to buy my domain name, but their two-step verification process is a bit annoying and will limit your number of edits in 2 hours.

Change your DNS settings for your domain.

<details>

<summary>If you use GoDaddy, you can follow the enclosed steps to edit DNS settings for your new github page.</summary>

1. Login to your GoDaddy account.
2. Click on the domain name you want to manage.
3. Click on the `DNS` tab.
4. Add a new [`A` record](https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site/managing-a-custom-domain-for-your-github-pages-site#configuring-an-apex-domain).

![A record](docs/A.jpg)

5. Add a new [`CNAME` record](https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site/managing-a-custom-domain-for-your-github-pages-site#configuring-a-subdomain).

![CNAME record](docs/CNAME.jpg)

</details>



Change the scripts/add-domain in `package.json` to your custom domain.
```txt
echo <your custom domain> > build/CNAME
```

Run the following command to deploy your website.

```bash
npm run deploy
```

Check the website at your custom domain.




## Common Questions

How to add a new project page and project image grid on the home page?

1. Create a new page js file in the `src/Pages/Project` folder.
2. Define the new project's constant in `src/Pages/Project/projects.const.js` for website routing.
3. Add a new project in the `src/Pages/Works/Works.js` file.

How to edit my about me page?

1. Edit the `src/Pages/About/About.js` file.

How to use my own resume?

1. Edit the `src/Pages/About/About.js` file.

```js
import cv from '../src/documents/<your resume name>.pdf';
```


How to edit my page content?

You can check `src/components` for my custom [styled-components](https://github.com/styled-components/styled-components).
You can also explore the components in [Ant Design](https://ant.design/components/overview/) and easily integrate them into your website.