Customizing a diffusion model for inpainting allows you to reconstruct specific parts of an image (the "masked" area) while keeping the rest of the image (the "known" area) intact and realistic. This is typically achieved by modifying the **Sampling Pipeline** (inference) rather than retraining the entire model.

---

## 1. The Mathematics of Diffusion
To understand the customization, we first need the two core processes of a Denoising Diffusion Probabilistic Model (DDPM).

### The Forward Process (Adding Noise)
In the forward process, we take a clean image $x_0$ and gradually add Gaussian noise over $T$ steps. The state at any time step $t$ can be calculated directly using the formula:

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$$

* $\bar{\alpha}_t$: A scheduler parameter that decreases over time.
* $\epsilon$: Random Gaussian noise $\sim \mathcal{N}(0, \mathbf{I})$.
* **Result:** As $t$ increases, the image becomes more noisy until it is pure white noise at $t=T$.

### The Reverse Process (Denoising)
The goal of the model (usually a UNet) is to learn to reverse this process. It predicts the noise $\epsilon_\theta$ present in $x_t$ to recover a slightly cleaner version $x_{t-1}$:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z$$

---

## 2. Step-by-Step Customization for Inpainting
Inpainting introduces a **Mask ($M$)**, where $M=1$ represents the region to be generated and $M=0$ represents the original pixels to be preserved. The customization happens inside the loop for each time step $t$.

### Step 1: Forward Noise Injection
For the current time step $t$, we take our **reference image** (the original image we want to keep) and apply the forward diffusion formula to get a noisy version of the known parts:
$$x_t^{known} = \text{ForwardNoise}(x_{ref}, t)$$

### Step 2: Composition (The Blending Step)
We combine the noisy known pixels from the reference image with the current generated "reverse" image ($x_t^{gen}$):
$$x_t^{combined} = (x_t^{known} \cdot (1 - M)) + (x_t^{gen} \cdot M)$$
* **$(1-M)$:** Selects the regions outside the mask (the context).
* **$M$:** Selects the region inside the mask (what the AI is drawing).

### Step 3: Denoising the Blended Image
We pass this combined image $x_t^{combined}$ into the model. Because the context pixels are "correct" (noisy versions of the real image), the model uses that context to predict how to denoise the masked area so that it blends seamlessly with the edges.
$$x_{t-1}^{gen} = \text{DenoiseStep}(x_t^{combined}, t)$$



---

## 3. What is Learnt and How?
The "learning" in this context refers to the initial training of the Diffusion Model, not the inpainting modification itself.

### What is Learnt?
The model learns the **Score Function** or the **Gradient of the Log-Density** of the data distribution. Essentially, it learns:
* **Structural Correlations:** How pixels relate to each other (e.g., if there is an eye, there is likely an eyebrow above it).
* **Texture and Realism:** What "natural" images look like vs. "noisy" images.
* **Conditional Probability:** During inpainting, the model uses its knowledge of the data distribution to find the most "likely" pixels to fill the mask, given the surrounding context pixels.

### How the Learning Happens
1.  **Dataset:** The model is trained on millions of images (e.g., LAION or ImageNet).
2.  **Objective Function:** The training uses a **Mean Squared Error (MSE)** loss between the actual noise added ($\epsilon$) and the noise predicted by the model ($\epsilon_\theta$):
    $$L_{simple} = E_{t, x_0, \epsilon} [ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 ]$$
3.  **Optimization:** Through backpropagation, the UNet's weights are adjusted to become experts at identifying and removing noise.

---

## Summary Table: Standard vs. Inpainting Sampling

| Feature | Standard Sampling | Inpainting Sampling |
| :--- | :--- | :--- |
| **Input** | Pure Noise | Pure Noise + Ref Image + Mask |
| **Step Logic** | Denoise $x_t$ to get $x_{t-1}$ | **Blend** Ref Noise with $x_t$, then denoise |
| **Constraint** | None (Free generation) | Constrained by original pixels $(1-M)$ |
| **Consistency** | High global consistency | High boundary consistency with existing image |


---

Here are the detailed answers to your questions regarding the mathematics and mechanics of diffusion models:

### 1. What is the scheduler parameter that decreases over time?
In diffusion models, the scheduler (or variance schedule) controls exactly how much noise is added to the image at each time step $t$. 

The parameter $\bar{\alpha}_t$ (alpha-bar) specifically represents the **proportion of the original image's signal** that remains intact. 
* At $t=0$ (the start), $\bar{\alpha}_t$ is very close to $1$, meaning the image is mostly clear.
* As $t$ approaches $T$ (the final step), $\bar{\alpha}_t$ gradually decreases toward $0$. 

Because $\bar{\alpha}_t$ decreases over time, the formula forces the original image signal to fade out, while the noise component correspondingly increases until the image becomes pure white noise.

### 2. What is $\mathbf{I}$ in Random Gaussian noise $\sim \mathcal{N}(0, \mathbf{I})$?
The $\mathbf{I}$ represents the **Identity Matrix**. 

When we say noise is sampled from a normal (Gaussian) distribution denoted as $\mathcal{N}(\mu, \Sigma)$, the first value is the mean ($\mu$) and the second is the covariance matrix ($\Sigma$). 
* A mean of **0** means the noise is centered around zero.
* A covariance of **$\mathbf{I}$** means that the noise added to every single pixel (and every color channel) is completely independent of the others, with a variance of $1$. 

In simpler terms, it guarantees that we are injecting standard, uncorrelated white noise evenly across the entire image.

### 3. Is each step $x_t$ in the forward process compared with $x_t$ in the reverse process?
**During training:** Yes, indirectly. The model takes a clean image, uses the forward process to instantly jump to step $x_t$, and then the neural network tries to predict the exact noise that was just added. The model's prediction is compared against the actual forward noise to calculate the error.
**During standard generation (inference):** No. The forward process isn't used at all. The model simply starts at pure noise ($x_T$) and steps backward.
**During inpainting (as described in the previous guide):** Yes! This is the core trick. At every step $t$, the algorithm calculates the forward noisy state $x_t$ of the original reference image and forcibly blends it with the reverse generating state $x_t$ so the model has the correct context to continue denoising.

### 4. Do forward noise injection and backward denoising happen one after the other?
For standard image generation, no. But for the **custom inpainting sampling loop**, yes. 

In the customized pipeline, the sequence inside the loop at time step $t$ happens consecutively:
1. **Forward:** Add noise to the reference background up to step $t$.
2. **Blend:** Combine this noisy background with the AI's current generated canvas.
3. **Backward:** Pass this blended image into the UNet to denoise it and step backward to $t-1$.
This cycle repeats sequentially until $t=0$.

### 5. During Inpainting, does the model use pixel data distribution to find the most likely pixels?
Yes, exactly. The model has no memory of what was actually behind the mask in your specific photo. 

Instead, it relies entirely on the **learned data distribution** from the millions of images it was trained on. It looks at the noisy unmasked pixels surrounding the hole and effectively calculates a conditional probability: *"Given the visual context of these surrounding edges, what textures, colors, and structures are statistically most likely to belong inside this gap based on all the natural images I've ever seen?"*

### 6. In the Objective function, is MSE calculated at each step and then added in the end to get the total MSE?
In mathematical theory, the objective function represents the expected loss across all time steps. However, **in actual computational practice during training, no.** Calculating the forward pass and predicting the noise for every single step from $t=1$ to $t=T$ for every image in a batch would require an impossible amount of memory and compute time. 

Instead, diffusion models use **random sampling**:
1. The system picks a batch of images.
2. For each image, it randomly selects just **one** time step $t$.
3. It calculates the Mean Squared Error (MSE) between the predicted noise and the actual noise for *just that single step*.
4. The neural network's weights are updated based on that one step.

By doing this repeatedly over thousands of batches and millions of iterations, the model effectively learns how to denoise at every possible step without having to sum them all up sequentially in a single pass.


---

Standard diffusion models fail at native inpainting because they are inherently designed for **unconstrained, global generation** rather than **constrained, local generation**. 

Without modifying the sampling pipeline (as discussed in the previous step) or retraining the model specifically for inpainting, a standard model runs into several critical roadblocks.

### 1. No Concept of a "Mask" or "Context"
A standard diffusion model's architecture (typically a UNet) is trained to take exactly two primary inputs: pure global noise and a text prompt. 
* It does not have input channels to accept a **Reference Image** or a **Binary Mask**. 
* Because it cannot "see" the boundaries of the hole you want to fill, it has no way to condition its generation on the surrounding pixels. It will simply try to generate a complete, standalone image from the noise.

### 2. Severe Boundary Inconsistencies (The "Seam" Problem)
Imagine you try a naive approach: you let the standard model generate a whole new image, cut out the piece you need, and paste it into the missing hole of your original image. 
* **Mismatch in Structure:** A line or shape ending at the edge of the mask will not seamlessly continue into the generated patch. 
* **Mismatch in Lighting and Color:** The generated patch might have a completely different color palette, lighting angle, or contrast level than the original image. 
* **Result:** You get a very obvious, jarring seam where the pasted patch meets the original image.

### 3. Independent Noise Trajectories
In a standard model, the denoising process happens uniformly across the entire canvas. The noise at step $t$ for the top left pixel is dependent on the global state of the image at step $t$. 

If you try to denoise *only* the masked area while leaving the known area as clean pixels, you create a massive mathematical clash. The neural network gets confused because it is seeing completely clean, sharp pixels right next to highly noisy pixels. It hasn't been trained to handle this stark contrast in a single forward pass, leading to distorted, artifact-heavy outputs near the edges.

### 4. Global Self-Attention Fails
Diffusion models use "attention mechanisms" to ensure the whole image makes sense together. If a standard model is forced to look at an image with a giant black or white box covering the missing area, the attention layers will treat that box as a literal object in the scene. It will try to generate things that interact with a giant black box, rather than understanding it needs to fill the space *behind* the box.

---

This is exactly why we either have to **hijack the sampling loop** (to constantly force the known pixels into the correct noisy state at every step to guide the model) or **train a dedicated inpainting model** from scratch that expects a mask and a masked image as inputs.

The inpainting capability is a transferable weight‑space difference, and adding this difference to any diffusion model mathematically equips it with correct mask‑aware generation.

---

**RePaint** is essentially a specific, high-end mathematical implementation of the **Inpainting** concept we discussed earlier. While "Inpainting" is the general task of filling in missing parts of an image, [RePaint](https://arxiv.org/pdf/2201.09865) is the formal research paper that perfected the "sampling-time trick" to make it work with any diffusion model.

Here is how they relate and where they differ:

### 1. The Core Logic is Identical
Both the [Medium guide](https://medium.com/@aromalma/how-to-customize-any-diffusion-models-for-inpainting-178111b239cd) you shared and the [RePaint paper](https://arxiv.org/pdf/2201.09865) use the same fundamental "Blending Step" at every time step $t$:
* **Keep the known:** Take the original image, add noise to it to match the current step ($x_t^{known}$).
* **Generate the unknown:** Let the AI predict the pixels for the masked area ($x_t^{unknown}$).
* **Combine:** Use the mask to stitch them together before moving to the next step.

### 2. The Difference: "Resampling" (The RePaint Special Sauce)
The main difference is that the basic inpainting method explained in the article is a **linear** process (it goes from $T \to 0$ once). The **RePaint** paper introduces a concept called **Resampling** to make the result much more realistic.

* **Standard Inpainting (Article):** Does one pass from noise to image. Sometimes this results in "seams" or "halos" at the edges of the mask because the generated part doesn't have enough time to harmonize with the background.
* **RePaint Strategy:** After the model denoises from step $t \to t-1$, RePaint intentionally adds a little bit of noise back to jump from $t-1 \to t$ again. It repeats this mini-loop several times at each stage.
* **The Result:** This "back-and-forth" allows the model to "look" at the boundary between the real pixels and the generated pixels multiple times, smoothing out the transition and ensuring the lighting and textures match perfectly.

### Summary Comparison

| Feature | Basic Inpainting (Article) | RePaint (Research Paper) |
| :--- | :--- | :--- |
| **Goal** | Fill a mask using a standard model. | Fill a mask using a standard model. |
| **Training** | None (Uses pre-trained model). | None (Uses pre-trained model). |
| **Process** | Linear ($T \to 0$). | Iterative/Resampling (Forward and Backward steps). |
| **Quality** | Good, but edges can be visible. | Superior, extremely seamless boundaries. |
| **Speed** | Fast. | Slower (due to resampling loops). |

In short: The article explains the **mechanism** (how to hijack the loop), while RePaint provides the **optimization** (how to make that hijacked loop produce professional-grade results).

Do you think the "resampling" (going back and forth in time) makes sense as a way to "smooth out" the edges of the mask?

---

## Metrics

### What is SSIM?
SSIM is a metric used to measure how similar two images are, outputting a score between **-1** (completely different) and **+1** (identical). Unlike older methods that simply check the absolute difference between individual pixels (like Mean Squared Error), SSIM is designed to mimic **human visual perception**. 

Humans are great at identifying structures in a scene. Therefore, SSIM compares two images (let's call the reference image $x$ and the sample image $y$) based on three key features: **Luminance**, **Contrast**, and **Structure**.

---

### Step 1: Comparing Luminance (Brightness)
Luminance measures the overall brightness of the image. Mathematically, it is calculated by finding the **mean** (average) of all the pixel values in an image, denoted by $\mu$.

To compare the luminance of image $x$ and image $y$, the formula is:
$$l(x, y) = \frac{2\mu_x\mu_y + C_1}{\mu_x^2 + \mu_y^2 + C_1}$$

* **In simple terms:** It compares the average brightness of the two images. 
* **What is $C_1$?** It is just a small constant added to the top and bottom to ensure the math doesn't break (i.e., avoiding dividing by zero if the means are zero).

### Step 2: Comparing Contrast
Contrast measures the spread or variance in the pixel values—how much the pixels differ from the average brightness. This is calculated using the **standard deviation**, denoted by $\sigma$.

To compare the contrast of image $x$ and image $y$, the formula is:
$$c(x, y) = \frac{2\sigma_x\sigma_y + C_2}{\sigma_x^2 + \sigma_y^2 + C_2}$$

* **In simple terms:** It checks if the dynamic range (the difference between the lightest and darkest pixels) is similar in both images.
* **What is $C_2$?** Similar to $C_1$, it is a small constant used for mathematical stability.

### Step 3: Comparing Structure
Structure looks at how the pixels relate to one another. To do this, the math first normalizes the images by dividing them by their standard deviation. Then, it uses **covariance**, denoted by $\sigma_{xy}$, to see how the pixel values in image $x$ and image $y$ change together.

The structure comparison formula is:
$$s(x, y) = \frac{\sigma_{xy} + C_3}{\sigma_x\sigma_y + C_3}$$

* **In simple terms:** It evaluates whether the patterns and shapes (the structure) align between the two images, regardless of their overall brightness or contrast.

### Step 4: The Final Combined SSIM Formula
To get the final SSIM score, we multiply the three comparison functions together. 
$$SSIM(x, y) = [l(x, y)]^\alpha \cdot [c(x, y)]^\beta \cdot [s(x, y)]^\gamma$$

If we assume that luminance, contrast, and structure are equally important ($\alpha = \beta = \gamma = 1$) and we set $C_3 = C_2/2$ to simplify things, the grand formula combines into:

$$SSIM(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

---

### Step 5: The "Plot Twist" (Local vs. Global SSIM)
The article notes that running this massive formula on an entire image at once (*globally*) doesn't actually yield the best results. Image statistics change drastically from one corner of a photo to another, and human eyes only focus on a small area in high resolution at any given time.

**The Solution:** Instead of calculating one giant score, the math is applied **locally**. 
1. An **11x11 Gaussian window** (think of it as a small magnifying glass that weights the center pixels more heavily than the edges) slides across the image pixel by pixel.
2. The $\mu$, $\sigma$, and SSIM formulas are calculated purely within that tiny window.
3. Finally, the algorithm takes the **mean** (average) of all those thousands of tiny, local SSIM scores to give you the final global SSIM value. This is known as the **Mean Structural Similarity Index**.

---

Let’s break down the **RePaint** paper. At its core, this paper tackles a common photo-editing problem with a brilliantly simple, yet powerful, new approach. 

Here is a detailed, easy-to-understand explanation of the concepts, followed by a step-by-step breakdown of the math.

---

### Part 1: The Big Picture (In Simple Terms)

**The Problem: Image Inpainting**
Image inpainting is the process of filling in missing or damaged parts of an image. Think of it like a smart digital eraser: you highlight a person you want to remove from the background, or a tear in an old photograph, and the AI fills in the hole so it looks natural. 

Traditionally, AI models (like GANs) are trained specifically to fill in certain types of holes (called "masks"). If you train an AI to fill in thin scratches, it will fail if you suddenly ask it to fill in a massive square hole. It also tends to just copy-paste surrounding textures rather than inventing semantically meaningful things (like correctly generating an eye if half a face is missing).

**RePaint's Solution**
The authors of RePaint asked: *What if we don't train a new AI to fill holes at all?* Instead, they took an off-the-shelf **Denoising Diffusion Probabilistic Model (DDPM)**. These are powerful AI models that have already been trained to generate beautiful, complete images from pure static (noise). 

RePaint cleverly "hijacks" the generation process of this standard AI. As the AI is creating a whole image from scratch, RePaint constantly steps in and says: *"Hey, overwrite this specific area with the pixels we already know from the original photo."* Because the AI already knows how to draw anything, and isn't tied to specific "hole shapes," RePaint can fill in any type of missing region—whether it's a thin line, a grid, or 75% of the entire image—with incredible realism.

---

### Part 2: How Diffusion Models Work

To understand the math, you first need to understand the two steps of a Diffusion Model:

1.  **The Forward Process (Destroying the Image):** Imagine taking a crisp photo and adding TV static (Gaussian noise) to it step-by-step until it looks like a screen of pure grey static. 
2.  **The Reverse Process (Generating the Image):** The AI is trained to reverse this. It looks at the static and learns to peel away the noise, step-by-step, until a clear image emerges. 

---

### Part 3: The Mathematics (Step-by-Step)

Don't let the Greek letters intimidate you; they are just shorthand for the concepts we just discussed.

Let’s define our terms:
* $x_0$: The final, perfect image (step 0).
* $x_t$: The image at step $t$ (which has some static/noise on it). 
* $x_T$: Pure static (the final step of adding noise).
* $m$: The "mask." This is a map where `1` means a pixel we *know* (original image), and `0` means a pixel that is *missing* (the hole).

#### 1. The Forward Process (Adding Noise)
When the AI is being trained, it learns how noise is added mathematically. To get the noisy image at step $t$ from the previous step $t-1$, we use this formula:

$$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$$

* **What this means:** The new noisy image ($x_t$) is drawn from a Normal/Gaussian distribution (the $\mathcal{N}$). We take the previous image ($x_{t-1}$), shrink its values slightly (using the $\beta_t$ schedule), and add some random static ($\beta_t I$).

We can also calculate what the image looks like at *any* timestep directly from the original image $x_0$:

$$q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)$$

* **What this means:** We can instantly figure out exactly how noisy the known parts of our image should be at step $t$ by adding a mathematically calculated chunk of noise ($(1-\bar{\alpha}_t)$) to the perfect original image ($x_0$).

#### 2. The Reverse Process (Denoising)
During generation, the AI (a neural network represented by $\theta$) tries to go backward. It looks at the noisy image $x_t$ and predicts the slightly less-noisy image $x_{t-1}$:

$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

* **What this means:** The AI predicts the mean ($\mu_\theta$) and the variance ($\Sigma_\theta$) of the noise, effectively "guessing" what the image looked like one step ago.

#### 3. RePaint's Brilliant Trick (Conditioning)
This is the core of the paper. RePaint doesn't let the AI just generate the whole image freely. At every single reverse step, it forces the known pixels to stay true to the original photo. 

It splits the image into two parts:

**A. The Known Part:**
We take the original photo's known pixels and artificially add the exact right amount of noise to match the current step:
$$x_{t-1}^{known} \sim \mathcal{N}(\sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)$$

**B. The Unknown Part (The Hole):**
We let the AI predict what should be in the hole based on its denoising skills:
$$x_{t-1}^{unknown} \sim \mathcal{N}(\mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

**C. Sticking Them Together:**
We use the mask $m$ to glue the real pixels and the AI's predicted pixels together into one image:
$$x_{t-1} = m \odot x_{t-1}^{known} + (1-m) \odot x_{t-1}^{unknown}$$
*(Note: The $\odot$ symbol just means multiplying pixel-by-pixel. If the mask $m$ is 1 (known), it keeps the known part and multiplies the unknown part by 0).*

#### 4. The Secret Sauce (Resampling)
If you only do Step 3, the image will look like a bad copy-paste job. The borders where the "known" pixels meet the "unknown" pixels will look jarring because the AI wasn't looking at the real pixels when it predicted the fake ones.

To fix this, RePaint does a "two steps forward, one step back" dance called **Resampling**:
After creating the glued-together image ($x_{t-1}$), the algorithm *intentionally adds noise back to it* to return to $x_t$:

$$x_t \sim \mathcal{N}(\sqrt{1-\beta_t}x_{t-1}, \beta_t I)$$

Then, it runs the denoising step again. By repeatedly blurring the glued image and re-denoising it, the AI is forced to look at the whole picture at once. This "harmonizes" the boundaries, melting the original pixels and the generated pixels together seamlessly. 

### Summary
RePaint is beautiful because of its simplicity. By carefully mathematically blending a photo's original known pixels with an AI's generated pixels during the "de-noising" phase—and repeatedly rubbing the edges together by adding and removing noise—it achieves stunning results without ever needing to be trained on how to fill a specific shape.

---

Peak Signal-to-Noise Ratio (**PSNR**) is a standard mathematical metric used to quantify the quality of a reconstructed or inpainted image compared to an original "ground truth" image. 

In the context of **Diffusion Inpainting** (like RePaint), PSNR measures how accurately the model filled in the missing pixels by comparing the generated output to what was originally there.

---

### 1. The Core Concept: Signal vs. Noise
Think of an image as a "signal." When a model tries to reconstruct an image, the differences between the original and the reconstructed version are treated as "noise."
* **Peak Signal:** The maximum possible strength of a pixel (e.g., pure white in a standard photo).
* **Noise:** The "errors" or deviations the model made while inpainting.

A **higher PSNR value** generally means the inpainted image is a closer match to the original. It is measured in **decibels (dB)**.

---

### 2. The Mathematics Step-by-Step

To calculate PSNR, you must first calculate the **Mean Squared Error (MSE)**.

#### Step A: Mean Squared Error (MSE)
MSE calculates the average squared difference between every pixel in the original image ($I$) and the inpainted image ($K$).
$$MSE = \frac{1}{m \times n} \sum_{i=0}^{m-1} \sum_{j=0}^{n-1} [I(i,j) - K(i,j)]^2$$
* **$m \times n$**: Total number of pixels.
* **$I - K$**: The difference in value for a specific pixel.
* **Squared ($^2$):** We square the difference so that "negative" errors (pixels that are too dark) and "positive" errors (pixels that are too bright) don't cancel each other out.



#### Step B: The PSNR Formula
Once you have the MSE, you plug it into the PSNR equation:
$$PSNR = 10 \cdot \log_{10} \left( \frac{MAX_I^2}{MSE} \right)$$
* **$MAX_I$**: The maximum possible pixel value. For a standard 8-bit image, this is **255**.
* **Logarithmic Scale ($\log_{10}$):** Because human perception of light is logarithmic and the range of errors can be huge, we use decibels to make the numbers easier to manage.

---

### 3. How it evaluates Diffusion Models (e.g., RePaint)
In a RePaint or Diffusion Inpainting task, the model is given an image with a "hole" (mask). The model uses a stochastic (random) diffusion process to "dream up" what belongs in that hole.

* **Reconstruction Accuracy:** PSNR tells us if the model's "dream" matches the actual reality of the original photo. If the original had a blue flower and the model painted a red one, the pixel values will differ wildly, leading to a high MSE and a **low PSNR**.
* **Typical Values:** * **> 30 dB:** Generally considered good quality (errors are hard to see).
    * **20–25 dB:** Acceptable, but visible "noise" or artifacts may be present.
    * **< 20 dB:** Poor quality; the inpainted area likely looks very different from the original.



---

### 4. The "Catch": Why PSNR isn't perfect
While PSNR is great for checking mathematical accuracy, it has a major flaw in evaluating AI models like Diffusion: **It doesn't care about "looking good."**

* **The Blurriness Trap:** A diffusion model might create a slightly blurry but mathematically "average" patch that gets a high PSNR. Meanwhile, it might create a sharp, beautiful, realistic object that is slightly shifted by 2 pixels; because those pixels don't align perfectly with the original, PSNR will give it a **very bad score**, even though a human would think it looks perfect.
* **Modern Alternative:** This is why researchers often use **LPIPS** (which measures "perceptual" similarity) or **FID** alongside PSNR to ensure the image looks realistic to humans.

---

While **PSNR** and **SSIM** focus on pixel-level accuracy and basic structure, **LPIPS** and **FID** are the "gold standards" for AI models because they evaluate images based on **deep learning features**. They basically use an AI's "brain" to judge another AI's work.

---

## 1. LPIPS (Learned Perceptual Image Patch Similarity)
LPIPS was created because humans often find "noisy" AI images more realistic than "blurry" images, even if the blurry ones have better pixel math. 

### How it works:
1.  **The "Judge":** LPIPS uses a pre-trained deep neural network (usually **VGG** or **AlexNet**) that has already learned to recognize objects and textures.
2.  **Feature Extraction:** Instead of comparing pixels, it feeds both the original and the generated image through this network.
3.  **Deep Comparison:** It compares the "activations" (the internal patterns the network sees) at different layers.
    * **Lower layers** detect edges and colors.
    * **Higher layers** detect complex textures and shapes.
4.  **The Score:** The difference between these features is calculated. 
    * **Lower Score (closer to 0):** The images are "perceptually" very similar.
    * **Higher Score:** The images look different to a human-like eye.

**In Diffusion Inpainting:** LPIPS is used to see if the filled-in area "feels" like it belongs to the same style and texture as the original, even if it isn't an exact pixel match.

---

## 2. FID (Fréchet Inception Distance)
While LPIPS compares two specific images, **FID** compares a **whole set** of generated images against a **whole set** of real images. It is the primary way we measure how "realistic" a model is overall.

### How it works:
1.  **The Inception Network:** It uses the "Inception-v3" model to extract features from thousands of real images and thousands of generated images.
2.  **The Distribution:** It calculates the mean ($\mu$) and covariance ($\Sigma$) of these features for both sets. Think of this as creating a "statistical fingerprint" for what "real" looks like versus what "AI-generated" looks like.
3.  **The Fréchet Distance:** It uses a specific math formula (the Fréchet distance) to measure the distance between these two fingerprints.
    * **Lower FID (closer to 0):** The generated images have the same variety and quality as the real ones.
    * **Higher FID:** The images are either low quality, repetitive (mode collapse), or look "fake."

**In Diffusion Inpainting:** Researchers use FID to ensure the model isn't just making "safe," blurry guesses, but is generating diverse, high-quality textures that match the statistics of real-world photos.

---

## Comparison Table: Which one to use?

| Metric | Level | What it actually measures | Goal |
| :--- | :--- | :--- | :--- |
| **PSNR** | Pixel | Exact mathematical error. | High accuracy. |
| **SSIM** | Local | Brightness, contrast, and structure. | High visual consistency. |
| **LPIPS** | Perceptual | Human-like "feel" and texture. | High realism per image. |
| **FID** | Statistical | Overall quality and diversity of the model. | High model performance. |

---

## How they are used for evaluation together
In a research paper for a model like **RePaint**, you will usually see a table comparing all four. 

* A model with **High PSNR but High LPIPS** is likely making blurry, "safe" images that don't look real. 
* A model with **Lower PSNR but Low LPIPS/FID** is likely creating sharp, realistic details that don't perfectly match the original (e.g., it drew a different type of leaf than the one that was masked out), but are considered "better" results by humans.


---